#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace std;
using fvec = vector<float>;
using fmat = vector<fvec>;

// ---------------- 超参 ----------------
const int VOCAB_SIZE = 256;
const int DIM = 64;        // 隐藏维度
const int N_HEAD = 4;      // 头数，DIM必须整除N_HEAD
const int N_LAYER = 2;     // Transformer层数
const int SEQ_LEN = 64;    // 最大上下文长度
const float EPS = 1e-5f;
const float MASK_NEG_INF = -1e9f;

mt19937 rng(42);
uniform_real_distribution<float> dist(-0.1f, 0.1f);

// 随机初始化权重
fmat rand_mat(int r, int c) {
    fmat w(r, fvec(c));
    for(auto &row : w) for(auto &x : row) x = dist(rng);
    return w;
}
fvec rand_vec(int n) {
    fvec v(n);
    for(auto &x : v) x = dist(rng);
    return v;
}

// 激活 GELU
float gelu(float x) {
    const float sqrt2 = sqrtf(2.0f);
    return 0.5f * x * (1.0f + tanhf(sqrt2 / M_PI * (x + 0.044715f * x*x*x)));
}

// softmax
void softmax(fvec &a) {
    float m = *max_element(a.begin(), a.end());
    float s = 0.0f;
    for(auto &x : a) {
        x = expf(x - m);
        s += x;
    }
    for(auto &x : a) x /= s;
}

// LayerNorm
void layernorm(fvec &x, const fvec &w, const fvec &b) {
    float mu = 0, var = 0;
    for(auto v : x) mu += v;
    mu /= x.size();
    for(auto v : x) var += (v-mu)*(v-mu);
    var /= x.size();
    float inv_std = 1.0f / sqrtf(var + EPS);
    for(int i=0;i<x.size();i++) {
        x[i] = w[i] * (x[i] - mu) * inv_std + b[i];
    }
}

// 矩阵乘法 A [n,k] @ B [k,m] -> C[n,m]
fmat matmul(const fmat &A, const fmat &B) {
    int n = A.size();
    int k = B.size();
    int m = B[0].size();
    fmat C(n, fvec(m, 0.0f));
    for(int i=0;i<n;i++){
        for(int t=0;t<k;t++){
            float av = A[i][t];
            if (av == 0) continue;
            for(int j=0;j<m;j++){
                C[i][j] += av * B[t][j];
            }
        }
    }
    return C;
}

// 向量乘矩阵 v [d] @ w [d, out]
fvec vec_mat(const fvec &v, const fmat &w) {
    int out = w[0].size();
    fvec res(out, 0.0f);
    for(int i=0;i<v.size();i++){
        float vi = v[i];
        if (vi == 0) continue;
        for(int j=0;j<out;j++){
            res[j] += vi * w[i][j];
        }
    }
    return res;
}

// 正弦位置编码 [seq_len, dim]
fmat pos_enc(int T, int d) {
    fmat pe(T, fvec(d, 0.0f));
    for(int t=0;t<T;t++){
        for(int i=0;i<d;i+=2){
            float div = powf(10000.0f, -(float)i / d);
            pe[t][i] = sinf(t * div);
            pe[t][i+1] = cosf(t * div);
        }
    }
    return pe;
}

// ====================== 【重点：标准因果多头注意力】 ======================
// x: [T, D] 输入序列
fmat mha(const fmat &x, const fmat &wq, const fmat &wk, const fmat &wv, const fmat &wo) {
    int T = x.size();
    int hd = DIM / N_HEAD;

    // 每一个位置单独算 q,k,v
    fmat q_all(T, fvec(DIM));
    fmat k_all(T, fvec(DIM));
    fmat v_all(T, fvec(DIM));
    for(int t=0;t<T;t++){
        q_all[t] = vec_mat(x[t], wq);
        k_all[t] = vec_mat(x[t], wk);
        v_all[t] = vec_mat(x[t], wv);
    }

    fmat out_seq(T, fvec(DIM, 0.0f));

    // 逐头计算
    for(int h=0;h<N_HEAD;h++){
        // 切出当前头 q,k,v: [T, hd]
        fmat q(T, fvec(hd));
        fmat k(T, fvec(hd));
        fmat v(T, fvec(hd));
        for(int t=0;t<T;t++){
            for(int i=0;i<hd;i++){
                q[t][i] = q_all[t][h*hd + i];
                k[t][i] = k_all[t][h*hd + i];
                v[t][i] = v_all[t][h*hd + i];
            }
        }

        // attention score = Q @ K^T / sqrt(hd)
        fmat att_score(T, fvec(T, 0.0f));
        for(int i=0;i<T;i++){
            for(int j=0;j<T;j++){
                float s = 0;
                for(int d=0;d<hd;d++) s += q[i][d] * k[j][d];
                att_score[i][j] = s / sqrtf((float)hd);
            }
        }

        // Causal Mask！上三角设置成 -inf，不能看到未来
        for(int i=0;i<T;i++){
            for(int j=i+1;j<T;j++){
                att_score[i][j] = MASK_NEG_INF;
            }
        }

        // 逐行softmax
        for(int i=0;i<T;i++){
            softmax(att_score[i]);
        }

        // attn @ V
        fmat head_out(T, fvec(hd, 0.0f));
        for(int i=0;i<T;i++){
            for(int j=0;j<T;j++){
                float a = att_score[i][j];
                for(int d=0;d<hd;d++){
                    head_out[i][d] += a * v[j][d];
                }
            }
        }

        // 拼回输出
        for(int t=0;t<T;t++){
            for(int i=0;i<hd;i++){
                out_seq[t][h*hd + i] = head_out[t][i];
            }
        }
    }

    // 输出投影 wo
    fmat final_out(T, fvec(DIM));
    for(int t=0;t<T;t++){
        final_out[t] = vec_mat(out_seq[t], wo);
    }
    return final_out;
}

// FFN
fmat ffn(const fmat &x, const fmat &w1, const fvec &b1, const fmat &w2, const fvec &b2) {
    int T = x.size();
    fmat h(T, fvec(w1[0].size()));
    for(int t=0;t<T;t++){
        h[t] = vec_mat(x[t], w1);
        for(int i=0;i<h[t].size();i++){
            h[t][i] = gelu(h[t][i] + b1[i]);
        }
    }
    fmat o(T, fvec(DIM));
    for(int t=0;t<T;t++){
        o[t] = vec_mat(h[t], w2);
        for(int i=0;i<o[t].size();i++){
            o[t][i] += b2[i];
        }
    }
    return o;
}

// Transformer Block
fmat block(fmat x,
           const fmat &wq,const fmat &wk,const fmat &wv,const fmat &wo,
           const fvec &ln1w, const fvec &ln1b,
           const fmat &w1,const fvec &b1,const fmat &w2, const fvec &b2,
           const fvec &ln2w, const fvec &ln2b)
{
    int T = x.size();
    // Pre-LN 结构（GPT标准）
    fmat x1 = x;
    for(int t=0;t<T;t++) layernorm(x1[t], ln1w, ln1b);
    fmat att = mha(x1, wq, wk, wv, wo);
    // 残差连接
    for(int t=0;t<T;t++){
        for(int d=0;d<DIM;d++) x[t][d] += att[t][d];
    }

    fmat x2 = x;
    for(int t=0;t<T;t++) layernorm(x2[t], ln2w, ln2b);
    fmat ff = ffn(x2, w1, b1, w2, b2);
    for(int t=0;t<T;t++){
        for(int d=0;d<DIM;d++) x[t][d] += ff[t][d];
    }
    return x;
}

struct Model {
    fmat token_emb;
    fmat pe;
    vector<fmat> wq,wk,wv,wo,w1,w2;
    vector<fvec> b1, b2, ln1w, ln1b, ln2w, ln2b;
    fmat lm_head;

    Model(){
        token_emb = rand_mat(VOCAB_SIZE, DIM);
        pe = pos_enc(SEQ_LEN, DIM);
        for(int l=0;l<N_LAYER;l++){
            wq.push_back(rand_mat(DIM, DIM));
            wk.push_back(rand_mat(DIM, DIM));
            wv.push_back(rand_mat(DIM, DIM));
            wo.push_back(rand_mat(DIM, DIM));
            w1.push_back(rand_mat(DIM, DIM*4));
            b1.push_back(rand_vec(DIM*4));
            w2.push_back(rand_mat(DIM*4, DIM));
            b2.push_back(rand_vec(DIM));
            ln1w.push_back(rand_vec(DIM));
            ln1b.push_back(rand_vec(DIM));
            ln2w.push_back(rand_vec(DIM));
            ln2b.push_back(rand_vec(DIM));
        }
        lm_head = rand_mat(DIM, VOCAB_SIZE);
    }

    // forward: 输入token列表，返回【最后一个位置】的logits，用来采样下一个token
    fvec forward(const vector<int> &tokens, float temp=0.8f) {
        int T = tokens.size();
        fmat x(T, fvec(DIM, 0.0f));
        // embedding + pos
        for(int t=0;t<T;t++){
            int id = tokens[t];
            for(int d=0;d<DIM;d++){
                x[t][d] = token_emb[id][d] + pe[t][d];
            }
        }
        // 所有层
        for(int l=0;l<N_LAYER;l++){
            x = block(x, wq[l],wk[l],wv[l],wo[l], ln1w[l],ln1b[l],
                      w1[l],b1[l],w2[l],b2[l], ln2w[l],ln2b[l]);
        }
        // 取最后一个token的hidden state
        fvec last = x.back();
        fvec logits = vec_mat(last, lm_head);
        for(auto &v : logits) v /= temp;
        return logits;
    }

    int sample(const fvec &logits) {
        fvec p = logits;
        softmax(p);
        uniform_real_distribution<float> s(0,1);
        float r = s(rng);
        float sum=0;
        for(int i=0;i<VOCAB_SIZE;i++){
            sum += p[i];
            if(sum >= r) return i;
        }
        return 0;
    }

    string generate(string prompt, int max_new=100, float temp=0.8f) {
        vector<int> tok;
        for(char c : prompt) tok.push_back((uint8_t)c);
        for(int i=0;i<max_new;i++){
            fvec logits = forward(tok, temp);
            int next_id = sample(logits);
            tok.push_back(next_id);
            prompt += (char)next_id;
            if(tok.size() >= SEQ_LEN) break;
        }
        return prompt;
    }
};

int main() {
    Model m;
    string prompt = "hi";
    string out = m.generate(prompt, 120, 0.7f);
    cout << out << endl;
    return 0;
}


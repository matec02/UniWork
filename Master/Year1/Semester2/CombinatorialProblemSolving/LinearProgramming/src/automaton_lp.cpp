#include <ilcplex/ilocplex.h>
#include <iostream>
#include <vector>
#include <unordered_map>

/* compile with: g++ -O2 -std=c++17 automaton_lp.cpp -o automaton_lp -I$CPLEX_HOME/include -I$CONCERT_HOME/include -L$CPLEX_HOME/lib/x86-64_linux/static_pic -L$CONCERT_HOME/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lpthread */

ILOSTLBEGIN

using namespace std;

constexpr size_t GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15ULL;


// Hash function for better searching for prefix, needed for vector<int>
struct VecHash {
    size_t operator()(const vector<int>& v) const noexcept {
        size_t h = 0;
        for (int b : v) h = (h << 1) ^ (b + GOLDEN_RATIO_64 + (h >> 2));
        return h;
    }
};

// Collecting all prefixes, empty one included
static void collectPrefixes(const vector<vector<int>>& sequence,
                            unordered_map<vector<int>,int,VecHash>& id,
                            vector<vector<int>>& list)
{
    auto enter=[&](const vector<int>& p){
        if(id.emplace(p,(int)id.size()).second){
            list.push_back(p); // if it is a new insertion add to list for preserving order
        } 
    };
    enter({}); // empty prefix
    for(const auto& w : sequence){ // build the HashMap with lambda func.
        vector<int> p;
        for(int bit : w){
            p.push_back(bit);
            enter(p);
        }
    }
}

static bool solveForN(int n,
                      const vector<vector<int>>& accept_seq,
                      const vector<vector<int>>& reject_seq,
                      const vector<vector<int>>& pref,
                      const unordered_map<vector<int>,int,VecHash>& idx,
                      vector<int>& T0,
                      vector<int>& T1,
                      vector<int>& acc_state)
{
    IloEnv   env;
    IloModel model(env);

    const int P = (int)pref.size();                   
    auto yPos = [=](int k,int s){ 
        return k*n + s; 
    }; // returns exactly the state

    // variables
    IloBoolVarArray y   (env, P*n); // prefix to state connection
    IloBoolVarArray t0  (env, n*n); // transition for 0
    IloBoolVarArray t1  (env, n*n); // transition for 1
    IloBoolVarArray acc (env, n); // acept or reject

    // exactly one state per prefix
    for(int i=0;i<P;++i){
        IloExpr sum(env);
        for(int j=0;j<n;++j){
            sum += y[yPos(i,j)];
        }
        model.add(sum == 1);
        sum.end();
    }

    // empty one is in state 0
    int idEps = idx.at({});
    model.add(y[yPos(idEps,0)] == 1);

    // only one succesor - deterministic
    for(int i=0;i<n;++i){
        IloExpr z0(env), z1(env);
        for(int j=0;j<n;++j){
            z0 += t0[i*n+j];
            z1 += t1[i*n+j];
        }
        model.add(z0 == 1);
        model.add(z1 == 1);
        z0.end(); 
        z1.end();
    }

    // if state i is followed by state j with some char c -> constraint
    for(const auto& p : pref){
        int idp = idx.at(p);
        for(int c=0;c<=1;++c){
            auto qvec = p;  
            qvec.push_back(c);
            auto it = idx.find(qvec);
            if(it==idx.end()) continue; // not needed
            int idq = it->second;
            for(int i=0;i<n;++i)
                for(int j=0;j<n;++j){
                    IloBoolVar t = (c==0 ? t0[i*n+j] : t1[i*n+j]);
                    // this transitition must be selected if i is followerd by j
                    model.add( y[yPos(idp,i)] + y[yPos(idq,j)] - t <= 1 );
                }
        }
    }


    auto forceLabel=[&](const vector<vector<int>>& S,bool accept){
        for(const auto& w : S){
            int k = idx.at(w);
            for(int s=0;s<n;++s){
                if(accept)
                    model.add( y[yPos(k,s)] - acc[s] <= 0 ); // this state must accept
                else
                    model.add( y[yPos(k,s)] + acc[s] <= 1 ); // this state must reject
            }
        }
    };
    forceLabel(accept_seq,true);
    forceLabel(reject_seq,false);

    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());
    if(!cplex.solve()){ 
        env.end(); 
        return false; 
    }

    // if there's a solution extract it
    T0.assign(n,-1); 
    T1.assign(n,-1); 
    acc_state.assign(n,0);
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            // all are binary so >0.5 works
            if(cplex.getValue(t0[i*n+j]) > 0.5) T0[i]=j;
            if(cplex.getValue(t1[i*n+j]) > 0.5) T1[i]=j;
        }
        acc_state[i]=(int)cplex.getValue(acc[i]);
    }
    env.end();
    return true;
}

vector<vector<int>> read_sequences(istream& in) {
    int num_sequences;
    in >> num_sequences;

    vector<vector<int>> sequences;

    for (int i = 0; i < num_sequences; ++i) {
        int num_bits;
        in >> num_bits;

        vector<int> bits;
        for (int j = 0; j < num_bits; ++j) {
            int bit;
            in >> bit;
            bits.push_back(bit);
        }
        sequences.push_back(bits);
    }

    return sequences;
}

int main(){
    
    vector<vector<int>> accept_lines = read_sequences(cin);
	vector<vector<int>> reject_lines = read_sequences(cin);
    
    unordered_map<vector<int>,int,VecHash> idx;
    vector<vector<int>> pref;
    collectPrefixes(accept_lines,idx,pref);
    collectPrefixes(reject_lines,idx,pref);

    // solving from 1 to prefix.size
    vector<int> T0,T1,acc_state;
    int bestN=-1;
    for(int n=1;n<=(int)pref.size();++n){
        if(solveForN(n,accept_lines,reject_lines,pref,idx,T0,T1,acc_state)){ 
            bestN=n; 
            break; 
        }
    }
    if(bestN<0){ 
        cerr<<"No DFA\n"; 
        return 1; 
    }

    // OUTPUT

    cout << accept_lines.size() << endl;
    for (const auto& seq : accept_lines) {
        cout << seq.size();
        if (!seq.empty()) cout << " ";
        for (size_t i = 0; i < seq.size(); ++i) {
            cout << seq[i];
            if (i < seq.size() - 1) cout << " ";
        }
        cout << endl;
    }

    cout << endl;

    cout << reject_lines.size() << endl;
    for (const auto& seq : reject_lines) {
        cout << seq.size();
        if (!seq.empty()) cout << " ";
        for (size_t i = 0; i < seq.size(); ++i) {
            cout << seq[i];
            if (i < seq.size() - 1) cout << " ";
        }
        cout << endl;
    }

    cout << bestN << endl;
    for(int s=0;s<bestN;++s){
        cout << T0[s] << ' ' << T1[s] << ' ' << acc_state[s] << endl;
    }
    return 0;
} 
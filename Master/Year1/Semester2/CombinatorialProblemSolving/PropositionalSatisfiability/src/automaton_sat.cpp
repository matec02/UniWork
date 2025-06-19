#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_set>

using namespace std;

#define V +

// string helpers
using literal = string;
using clause = string;

// for negating literals
literal operator-(const literal &lit)
{
    return (lit[0] == '-') ? lit.substr(1) : ("-" + lit);
}

int n;
int n_vars;
long long n_clauses;

vector<vector<int>> seq_bits;
vector<int> seq_accept;
vector<long long> seq_offset;

ofstream cnf;

// DIMACs integers

inline int id_T(int i, int b, int j) { return i * 2 * n + b * n + j + 1; }
inline int id_A(int i) { return 2 * n * n + i + 1; }
inline int id_S(int s, int p, int i)
{
    return 2 * n * n + n + static_cast<int>(seq_offset[s] + p * n + i + 1);
}

inline literal lit(int id) { return to_string(id) + " "; }
inline literal T(int i, int b, int j) { return lit(id_T(i, b, j)); }
inline literal A(int i) { return lit(id_A(i)); }
inline literal S(int s, int p, int i) { return lit(id_S(s, p, i)); }

// helpers CNF

void add_clause(const clause &c)
{
    cnf << c << "0\n";
    ++n_clauses;
}

void add_amo(const vector<literal> &z)
{
    const int N = static_cast<int>(z.size());
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            add_clause(-z[i] V - z[j]);
}

void add_exactly_one(const vector<literal> &z)
{
    clause c;
    for (const auto &lit : z)
        c = c V lit;
    add_clause(c);
    add_amo(z);
}

// writing CNF
void write_CNF()
{
    n_clauses = 0;

    // deterministic solver
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < 2; ++b)
        {
            vector<literal> z;
            for (int j = 0; j < n; ++j)
                z.push_back(T(i, b, j));
            add_exactly_one(z);
        }

    // sequence
    const int S_cnt = static_cast<int>(seq_bits.size());
    for (int s = 0; s < S_cnt; ++s)
    {
        const auto &bits = seq_bits[s];
        const int m = static_cast<int>(bits.size());

        // one state per prefix
        for (int p = 0; p <= m; ++p)
        {
            vector<literal> z;
            for (int i = 0; i < n; ++i)
                z.push_back(S(s, p, i));
            add_exactly_one(z);
        }

        // initial state
        add_clause(S(s, 0, 0));

        // adding clauses pr transition
        for (int p = 0; p < m; ++p)
        {
            int b = bits[p];
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    add_clause(-S(s, p, i) V - T(i, b, j) V S(s, p + 1, j));
        }

        // accept or rject
        for (int i = 0; i < n; ++i)
        {
            if (seq_accept[s])
                add_clause(-S(s, m, i) V A(i));
            else
                add_clause(-S(s, m, i) V - A(i));
        }
    }

    cnf << "p cnf " << n_vars << " " << n_clauses << "\n";
}

vector<vector<int>> read_sequences(istream &in)
{
    int num_sequences;
    in >> num_sequences;

    vector<vector<int>> sequences;

    for (int i = 0; i < num_sequences; ++i)
    {
        int num_bits;
        in >> num_bits;

        vector<int> bits;
        for (int j = 0; j < num_bits; ++j)
        {
            int bit;
            in >> bit;
            bits.push_back(bit);
        }
        sequences.push_back(bits);
    }

    return sequences;
}

int main()
{
    vector<vector<int>> accept_lines = read_sequences(cin);
    vector<vector<int>> reject_lines = read_sequences(cin);

    const int S_cnt = accept_lines.size() + reject_lines.size();
    seq_bits.reserve(S_cnt);
    seq_accept.reserve(S_cnt);
    for (const auto &s : accept_lines)
    {
        seq_bits.push_back(s);
        seq_accept.push_back(1);
    }
    for (const auto &s : reject_lines)
    {
        seq_bits.push_back(s);
        seq_accept.push_back(0);
    }

    // upper bound for states - distinct prefixes +1
    int max_states = 0;
    {
        unordered_set<string> pref_seen;
        for (const auto &seq : seq_bits)
        {
            string key;
            pref_seen.insert(key);
            for (int b : seq)
            {
                key += char('0' + b);
                pref_seen.insert(key);
            }
        }
        max_states = static_cast<int>(pref_seen.size());
        if (max_states < 1)
            max_states = 1;
    }

    // iterative search
    for (n = 1; n <= max_states; ++n)
    {
        seq_offset.resize(S_cnt);
        long long off = 0, totalS = 0;
        for (int s = 0; s < S_cnt; ++s)
        {
            seq_offset[s] = off;
            long long chunk = ((long long)seq_bits[s].size() + 1) * n;
            off += chunk;
            totalS += chunk;
        }

        n_vars = 2 * n * n + n + static_cast<int>(totalS);

        cnf.open("tmp.rev");
        write_CNF();
        cnf.close();

        // kissat run
        const string cmd =
            "tac tmp.rev | kissat | grep -E -v '^c' | cut --delimiter=' ' --field=1 --complement > tmp.out";
        system(cmd.c_str());

        ifstream sol("tmp.out");
        string result;
        sol >> result;
        // if solution is not found continue searching
        if (result != "SATISFIABLE")
        {
            sol.close();
            continue;
        }

        // decode solution
        vector<vector<int>> delta(n, vector<int>(2, -1));
        vector<int> acc(n, 0);
        int lit_int;
        while (sol >> lit_int)
        {
            if (lit_int <= 0)
                continue;
            if (lit_int <= 2 * n * n)
            {
                int tmp = lit_int - 1;
                int i = tmp / (2 * n);
                int rem = tmp % (2 * n);
                int b = rem / n;
                int j = rem % n;
                delta[i][b] = j;
            }
            else if (lit_int <= 2 * n * n + n)
            {
                int i = lit_int - (2 * n * n) - 1;
                acc[i] = 1;
            }
        }
        sol.close();

        // output
        cout << accept_lines.size() << endl;
        for (const auto &seq : accept_lines)
        {
            cout << seq.size();
            if (!seq.empty())
                cout << " ";
            for (size_t i = 0; i < seq.size(); ++i)
            {
                cout << seq[i];
                if (i < seq.size() - 1)
                    cout << " ";
            }
            cout << endl;
        }

        cout << endl;

        cout << reject_lines.size() << endl;
        for (const auto &seq : reject_lines)
        {
            cout << seq.size();
            if (!seq.empty())
                cout << " ";
            for (size_t i = 0; i < seq.size(); ++i)
            {
                cout << seq[i];
                if (i < seq.size() - 1)
                    cout << " ";
            }
            cout << endl;
        }

        cout << endl
             << n << endl;
        for (int i = 0; i < n; ++i)
            cout << delta[i][0] << ' ' << delta[i][1] << ' ' << acc[i] << endl;

        return 0;
    }

    cerr << "No DFA found" << endl;
    return 1;
}

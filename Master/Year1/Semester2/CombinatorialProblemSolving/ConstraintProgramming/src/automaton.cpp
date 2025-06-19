#include <cassert>
#include <cstdlib>
#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace Gecode;

class Automaton : public Space {
private:
	BoolVarArray accepted;
	IntVarArray reading_zero;
	IntVarArray reading_one;

    void simulate_sequence(const vector<bool>& bits, int no_states, int expected_accept) {
        int len = bits.size();
        IntVarArray f(*this, len + 1, 0, no_states - 1);

        rel(*this, f[0] == 0);

        for (int i = 0; i < len; ++i) {
            if (bits[i] == 0)
                element(*this, reading_zero, f[i], f[i + 1]);
            else
                element(*this, reading_one, f[i], f[i + 1]);
        }

        BoolVar final_accept(*this, 0, 1);
        element(*this, accepted, f[len], final_accept);
        rel(*this, final_accept == expected_accept);

        branch(*this, reading_zero, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
        branch(*this, reading_one, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
        branch(*this, accepted, BOOL_VAR_NONE(), BOOL_VAL_MIN());
    }

public:
	Automaton(int no_states, vector<vector<bool>> accept_lines, vector<vector<bool>> reject_lines) :
        accepted(*this, no_states, 0, 1), 
        reading_zero(*this, no_states, 0, no_states-1), 
        reading_one(*this, no_states, 0, no_states-1) {

        for (const auto& bits : accept_lines) {
            simulate_sequence(bits, no_states, 1);
        }

        for (const auto& bits : reject_lines) {
            simulate_sequence(bits, no_states, 0);
        }
	}

    Automaton(Automaton& s) : Space(s) {
        accepted.update(*this, s.accepted);
        reading_zero.update(*this, s.reading_zero);
        reading_one.update(*this, s.reading_one);
    }

    virtual Space* copy(void) {
        return new Automaton(*this);
    }

    void print() const {
        int n = accepted.size();
        cout << n << endl;
        for (int i = 0; i < n; ++i) {
            cout << reading_zero[i].val() << " "
                << reading_one[i].val() << " "
                << accepted[i].val();
            if (i!=n-1){
                cout << endl;
            }
        }
    }

};

vector<vector<bool>> read_sequences(istream& in) {
    int num_sequences;
    in >> num_sequences;

    vector<vector<bool>> sequences;

    for (int i = 0; i < num_sequences; ++i) {
        int num_bits;
        in >> num_bits;

        vector<bool> bits;
        for (int j = 0; j < num_bits; ++j) {
            int bit;
            in >> bit;
            bits.push_back(static_cast<bool>(bit));
        }
        sequences.push_back(bits);
    }

    return sequences;
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cerr << "Wrong command line." << endl;
		return 1;
	}

	ifstream input(argv[1]);
	if (!input) {
		cerr << "Error with reading file" << endl;
		return 1;
	}

	
	vector<vector<bool>> accept_lines = read_sequences(input);
	vector<vector<bool>> reject_lines = read_sequences(input);


    // for output
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
    
    auto start_time = std::chrono::high_resolution_clock::now();

    int n = 1;
    Automaton* solution = nullptr;
    cout << endl;
    //bool timeout = false;

    while (true) {

        Automaton* a = new Automaton(n, accept_lines, reject_lines);
        DFS<Automaton> dfs(a);
        //delete a;

        solution = dfs.next();
        if (solution != nullptr) {
            break;
        }
        n++;


    }

    /*if (timeout) {
        std::cout << "No solution found within 60 seconds" << std::endl;
        // Clean up any remaining resources
        return 1;
    }*/

    solution->print();
    delete solution;

    /*auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Print the elapsed time
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;*/

    return 0;
}
import itertools
import sys

# Korištene prezentacije:
# UUUI / 5. Prikazivanje znanja formalnom logikom,
# UUUI / 6. Automatsko zakljucivanje


# Klasa za generirane klauzale i pamcenje njihovih "roditelja"
class GeneratedClause:
    def __init__(self, parent1, parent2, clause):
        self.parent1 = parent1
        self.parent2 = parent2
        self.clause = clause


# Funkcija za uklananje redundantnih klauzula
def remove_subsumed_clause(clauses):
    to_remove = set()
    for clause1, clause2 in itertools.combinations(clauses, 2):
        # Ispitivanje je li jedna podskup druge
        if clause1 < clause2:
            to_remove.add(clause2)
        elif clause2 < clause1:
            to_remove.add(clause1)
    clauses.difference_update(to_remove)
    return clauses


# Uklananje nevažnih klauzula - tautologija
def check_tautology(clause):
    add_clause = True
    # Iteriranje po kombinacijama literala u klauzuli
    for literal1, literal2 in itertools.combinations(clause, 2):
        if literal1 == opposite_literal(literal2):
            add_clause = False
            break
    return add_clause


# Pomocna funkcija za pretvaranje u komplementarni literal
def opposite_literal(literal):
    if literal[0] == "~":
        return literal[1:]
    else:
        return "~" + literal


# Funkcija koja razrjesava roditeljske klauzule i vraca skup rezolventi
def pl_resolve(clause1, clause2, visited, final_clauses, all_used_clauses):
    new_clauses = set()
    for literal_clause in clause1:
        # Pronalazak roditeljskih klauzula ako vec nisu obradene
        if opposite_literal(literal_clause) in clause2 and clause1.union(clause2) not in visited:
            # Ako su obe klauzule sastavljene od jednog literala onda je rezolvent samo "NIL"
            if len(clause1) == 1 and len(clause2) == 1:
                all_used_clauses.append(GeneratedClause(clause1, clause2, "NIL"))
                new_clauses.add("NIL")
            else:
                # Stvaranje nove klauzule literala po kojem se izvodi rezolucija
                new_clause = frozenset(
                    literal for literal in clause1.union(clause2)
                    if literal != literal_clause and literal != opposite_literal(literal_clause)
                )
                # Provjerava da nova nastala klauzula nije tautologija
                if check_tautology(new_clause):
                    # Provjerava da nova nastala klauzula nije nadskup neke vec dobivene prijasnje klauzule
                    subsumed_cluase = False
                    for clause in final_clauses:
                        if new_clause > clause:
                            subsumed_cluase = True
                            break
                    if not subsumed_cluase:
                        # Ako nije dodaje se u skup rezolventi
                        all_used_clauses.append(GeneratedClause(clause1, clause2, new_clause))
                        new_clauses.add(new_clause)
            # Dodavanje u skup para roditeljskih klauzula kako se ne bi ponovile
            visited.add(clause1.union(clause2))
    # Vracanje skupa rezolventi
    return new_clauses


# Funkcija koja predstavlja algoritam rezolucije opovrgavanjem
def pl_resolution(clauses, final_clauses, all_used_clauses):
    # Inicijalizacija mogucih novih rezovenata i skupa posjecenih roditeljskih klauzula
    new = set()
    visited = set()
    while True:
        # Odabir klauzula iz premisa i SoS skupa
        for clause in clauses:
            for final_clause in final_clauses:
                # Poziv funkcije koja razrjesava roditeljske klauzule
                resolvents = pl_resolve(clause, final_clause, visited, final_clauses, all_used_clauses)
                # Ako je pronaden NIL vraca se True
                if "NIL" in resolvents:
                    return True
                # Dodavanje skupa novih rezolvenata jedne iteracija u skup svih novih
                new = new.union(resolvents)

        # Odabir klauzula direktno unutar SoS skupa
        for clause1, clause2 in itertools.combinations(final_clauses, 2):
            resolvents = pl_resolve(clause1, clause2, visited, final_clauses, all_used_clauses)
            if "NIL" in resolvents:
                return True
            new = new.union(resolvents)

        # Ako je skup svih novih rezolvenata podskup SoS skupa pretrazivanje prestaje
        # jer ne dobivamo nove razlicite klauzule
        if new.issubset(clauses.union(final_clauses)):
            return False
        # Dodavanje klauzula u SoS skup
        final_clauses = final_clauses.union(new)


# Funkcija za formirani ispis frozeset-a
def format_frozenset(frozen_set):
    if 'NIL' in frozen_set:
        return 'NIL'
    else:
        return ' v '.join(str(element) for element in frozen_set)


# Funkcija za ispisivanje koristenih premisa i ciljnjih stanja
def print_premises_and_final_state(clauses, final_clauses):
    i = 1
    indexed_list = []
    for clause in clauses:
        clause_string = " v ".join(str(literal) for literal in clause)
        indexed_list.append(GeneratedClause(None, None, clause))
        print(f"{i}. {clause_string}")
        i += 1
    for clause in final_clauses:
        clause_string = " v ".join(str(literal) for literal in clause)
        print(f"{i}. {clause_string}")
        i += 1
        indexed_list.append(GeneratedClause(None, None, clause))
    print("==================")
    return indexed_list


# Funkcija za konstruiranje puta od zavrsne klauzule NIL
def trace_back(generated_clause, premises, final_clauses, all_used_clauses,
               checked_clauses, used_premises, used_final_clauses, checked_objects_of_clauses):

    # Dodavanje koristenih premisa
    for parent in (generated_clause.parent1, generated_clause.parent2):
        if parent in premises:
            used_premises.add(parent)
        if parent in final_clauses:
            used_final_clauses.add(parent)

    # Razlog za zaustavljanje rekurzije
    if ((generated_clause.parent1 in premises and generated_clause.parent2 in final_clauses)
            or (generated_clause.parent1 in final_clauses and generated_clause.parent2 in premises)
            or (generated_clause.parent1 in final_clauses and generated_clause.parent2 in final_clauses)):
        return

    for used_clause in all_used_clauses:
        if used_clause == generated_clause:
            continue

        # Poziv rekurzije za klauzulu koja je roditelj ispitivane klauzule
        if used_clause.clause not in checked_clauses:
            for parent in (generated_clause.parent1, generated_clause.parent2):
                if parent == used_clause.clause:
                    # Pamcenje vec isptivane klauzule
                    checked_clauses.append(used_clause.clause)
                    checked_objects_of_clauses.append(used_clause)
                    trace_back(used_clause, premises, final_clauses, all_used_clauses, checked_clauses, used_premises,
                               used_final_clauses, checked_objects_of_clauses)


# Funkcija za formitirani ispis koristenih premisa i ciljnih stanje
# te dobivenih novih klauzula uz indeks njegovih roditelja
def resolution_with_print(clauses, final_clauses, all_used_clauses):
    checked_clauses, checked_objects_of_clauses, used_premises, used_final_clouses = [], [], set(), set()
    # Dobivanje koristenih premisa, ciljnjih stanja i novih klauzula
    trace_back(all_used_clauses[-1], clauses, final_clauses, all_used_clauses, checked_clauses,
               used_premises, used_final_clouses, checked_objects_of_clauses)
    # Prvotno ispisivanje koristenih premisa i ciljnjih stanja
    indexed_list = print_premises_and_final_state(used_premises, used_final_clouses)
    checked_objects_of_clauses.append(all_used_clauses[-1])
    # Za svaku generiranu klauzaulu gleda se je li se koristila
    # ako je dodaje se u index listu koja sluzi za pracenje indeksa klauzula
    for used_clause in all_used_clauses:
        if used_clause in checked_objects_of_clauses:
            index_parents = []
            # Trazenje roditelja generirane klauzule
            for added_clause in indexed_list:
                if used_clause.parent1 == added_clause.clause:
                    index_parents.append(indexed_list.index(added_clause) + 1)
                if used_clause.parent2 == added_clause.clause:
                    index_parents.append(indexed_list.index(added_clause) + 1)
            index_parents.sort()
            print(f"{len(indexed_list)+1}. {format_frozenset(used_clause.clause)} "
                  f"({index_parents[0]}, {index_parents[1]})")
            indexed_list.append(used_clause)
    print("==================")
    return


def main():
    resolution_txt = ""
    cooking_txt = ""
    cooking_input = ""

    # Parsiranje inputa
    for i in range(len(sys.argv)):
        if sys.argv[i] == "resolution":
            resolution_txt = sys.argv[i + 1]
        elif sys.argv[i] == "cooking":
            cooking_txt = sys.argv[i + 1]
            cooking_input = sys.argv[i + 2]

    # Inicijalizacija skupa premisa
    clauses = set()
    if resolution_txt or cooking_txt:
        # Ovisno koja je naredba citanje pripadajuce datoteke
        if resolution_txt:
            with open(resolution_txt, "r") as f:
                file = f.readlines()
            range_loop = len(file) - 1
        else:
            with open(cooking_txt, "r") as f:
                file = f.readlines()
            range_loop = len(file)
        # Parsiranje za ciljna stanja
        final_clause_string = file[len(file) - 1].strip().lower()
        final_clause_list = file[len(file) - 1].strip().lower().split(" v ")
        final_clauses = set()
        # Negiranje ciljnog stanja
        for literal in final_clause_list:
            if "~" in literal:
                final_clauses.add(frozenset(literal.replace("~", "")))
            else:
                final_clauses.add(frozenset(["~" + literal]))

        all_used_clauses = []
        # Dodavanje klauzula u skup premisa
        for i in range(range_loop):
            if not file[i].startswith("#"):
                clause = frozenset(file[i].strip().lower().split(" v "))
                if check_tautology(clause):
                    clauses.add(clause)

        # Uklananje redundantnih klauzula
        remove_subsumed_clause(clauses)

        # Ako je pozvana samo rezolucija
        if resolution_txt:
            if pl_resolution(clauses, final_clauses, all_used_clauses):
                resolution_with_print(clauses, final_clauses, all_used_clauses)
                print(f"[CONCLUSION]: {final_clause_string} is true")
            else:
                print(f"[CONCLUSION]: {final_clause_string} is unknown")
        # Ako je pozvana kuharica
        if cooking_txt:
            # Citanje datoteke s naredbama
            with open(cooking_input, "r") as f:
                file_input = f.readlines()
            for line in file_input:
                task = line.strip().lower()
                indetifactor_task = task[-1]
                literals = task[:-2]
                # Ovisno o naredbi izvodenje odredenog zadatka
                # Ako je upitnik onda rezolucija s zadanim ciljnjim stanjem
                if indetifactor_task == '?':
                    # Isti postupak kao samo kod rezolucije
                    final_clause_list = literals.split(" v ")
                    final_clauses = set()
                    for literal in final_clause_list:
                        if "~" in literal:
                            final_clauses.add(frozenset(literal.replace("~", "")))
                        else:
                            final_clauses.add(frozenset(["~" + literal]))
                    all_used_clauses = []
                    remove_subsumed_clause(clauses)
                    if pl_resolution(clauses, final_clauses, all_used_clauses):
                        resolution_with_print(clauses, final_clauses, all_used_clauses)
                        print(f"[CONCLUSION]: {literals} is true\n")
                    else:
                        print(f"[CONCLUSION]: {literals} is unknown\n")
                # Dodaje se nova premisa ako je naredba +
                elif indetifactor_task == '+':
                    new_clause = frozenset(literals.split(" v "))
                    # Provjerava je li uopće vazna odnosno je li tautologija
                    if check_tautology(new_clause):
                        clauses.add(new_clause)
                # Uklanja se premisa ako je naredba -
                elif indetifactor_task == '-':
                    new_clause = frozenset(literals.split(" v "))
                    if check_tautology(new_clause):
                        clauses.remove(new_clause)


if __name__ == "__main__":
    main()

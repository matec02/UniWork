import argparse
import heapq

# Korištene prezentacije: UUUI / 2.Pretraživanje prostora stanja, UUUI / 3.Heurističko pretraživanje
def bfs(pocetno_stanje, cvorovi, ciljna_stanja):
    # Inicijalizacija prioritetnog reda otvorenog za frontu te liste posjecenih stanje, u ovom slucaju rjecnik
    otvoreni = []
    visited = {}
    # Inicijalno dodavanje pocetnog stanja u frontu
    heapq.heappush(otvoreni, pocetno_stanje)
    broj_posjecenih = 0
    while len(otvoreni) > 0:
        # Uzimanje prvog elementa s fronte
        n = heapq.heappop(otvoreni)
        # Uvecavamo broj posjecenih stanja
        broj_posjecenih += 1
        # Ako je to stanje zapravo ciljno zavrsavamo pretragu te vracamo to stanje, broj posjecenih stanje i rjecnik
        # posjecena stanja
        if n[1] in ciljna_stanja:
            return n, broj_posjecenih, visited
        # Ako nije ciljno stanje nastavljamo dalje te ga dodajemo u rjecnik posjecenih stanja te kao kljuc stavljamo
        # stanje, rjecnik ne moze imati dva kljuca istog naziva -> samo jednom mozemo doci u stanje
        # Rjecnik oblika: {stanje: (roditelj, dubina, udaljenost od roditelja), ...}
        visited[n[1]] = (n[2], n[0], n[3])
        # Dobavljanje svih sljedbenika stanja koje obradujemo
        rjecnik_prijelaza = cvorovi.get(n[1])
        for sljedbenik, udaljenost in rjecnik_prijelaza.items():
            # Dodavanje sljedebnika u frontu ako vec nisu posjeceni te uvecavanje njihove dubine za 1 od dubine
            # stanja koje se obraduje
            if sljedbenik not in visited:
                nova_vrijednost = n[0] + 1
                # Priroriteni red sortira po prvom elementu tuple-a (dubina), ako je ista dubina onda po drugom (stanje)
                # odnono abecedno
                heapq.heappush(otvoreni, (nova_vrijednost, sljedbenik, n[1], udaljenost))
    return False, None, None


def ucs(pocetno_stanje, cvorovi, ciljna_stanja):
    # Inicijalizacija prioritetnog reda otvorenog za frontu te liste posjecenih stanje, u ovom slucaju rjecnik
    otvoreni = []
    visited = {}
    # Inicijalno dodavanje pocetnog stanja u frontu
    heapq.heappush(otvoreni, pocetno_stanje)
    broj_posjecenih = 0
    while len(otvoreni) > 0:
        # Uzimanje prvog elementa s fronte
        n = heapq.heappop(otvoreni)
        # Uvecavamo broj posjecenih stanja
        broj_posjecenih += 1
        # Ako je to stanje zapravo ciljno zavrsavamo pretragu te vracamo to stanje, broj posjecenih stanje i rjecnik
        # posjecena stanja
        if n[1] in ciljna_stanja:
            return n, broj_posjecenih, visited
        # Ako nije ciljno stanje nastavljamo dalje te ga dodajemo u rjecnik posjecenih stanja te kao kljuc stavljamo
        # stanje, rjecnik ne moze imati dva kljuca istog naziva -> samo jednom mozemo doci u stanje
        # Rjecnik oblika: {stanje: (roditelj, trenutacna udaljenost), ...}
        visited[n[1]] = (n[2], n[0])
        # Dobavljanje svih sljedbenika stanja koje obradujemo
        rjecnik_prijelaza = cvorovi.get(n[1])
        # Dodavanje sljedebnika u frontu ako vec nisu posjeceni te uvecavanje njihove udaljenosti od pocetnog stanja
        # za udaljenost prijelaza iz stanje koje obradujemo do njegovog sljedbenika
        for sljedbenik, udaljenost in rjecnik_prijelaza.items():
            if sljedbenik not in visited:
                nova_vrijednost = n[0] + udaljenost
                # Priroriteni red sortira po prvom elementu tuple-a (trenutacna udaljenost od poc. stanja),
                # ako je ista udaljenost onda po drugom (stanje) odnono abecedno
                heapq.heappush(otvoreni, (nova_vrijednost, sljedbenik, n[1]))
    return False, None, None


def astar(pocetno_stanje, cvorovi, ciljna_stanja, heuristika):
    # Inicijalizacija prioritetnog reda otvorenog za frontu te liste posjecenih stanje, u ovom slucaju rjecnik
    otvoreni = []
    visited = {}
    # Inicijalno dodavanje pocetnog stanja u frontu
    heapq.heappush(otvoreni,  pocetno_stanje)
    broj_posjecenih = 0
    while len(otvoreni) > 0:
        # Uzimanje prvog elementa s fronte
        n = heapq.heappop(otvoreni)
        # Uvecavamo broj posjecenih stanja
        broj_posjecenih += 1
        # Ako je to stanje zapravo ciljno zavrsavamo pretragu te vracamo to stanje, broj posjecenih stanje i rjecnik
        # posjecena stanja
        if n[1] in ciljna_stanja:
            return n, broj_posjecenih, visited
        # Ako nije ciljno stanje nastavljamo dalje te ga dodajemo u rjecnik posjecenih stanja te kao kljuc stavljamo
        # stanje, rjecnik ne moze imati dva kljuca istog naziva -> samo jednom mozemo doci u stanje
        # Rjecnik oblika: {stanje: (roditelj, trenutacna udaljenost), ...}
        visited[n[1]] = (n[2], n[3])
        # Dobavljanje svih sljedbenika stanja koje obradujemo
        rjecnik_prijelaza = cvorovi.get(n[1])
        # Ako je sljedbenik vec u listi otvorenih ili posjecenih (zatvorenih) stanja usporedujemo njegovu udaljenost
        # od pocetnog stanja sa odgovarajucim stanjem u listi otvorenih ili posjecnih stanja, ako je ono manje onda
        # izbacujemo odgovarajuce stanje iz njegovo liste te dodajemo sljedbenika u frontu, ako je medutim vece onda
        # zanemarujemo i nastavljamo sa sljedecim sljedebnikom
        for sljedbenik, udaljenost in rjecnik_prijelaza.items():
            insert = True
            if sljedbenik in visited:
                if visited[sljedbenik][1] < n[3] + udaljenost:
                    insert = False
                else:
                    del visited[sljedbenik]
            else:
                for cvor in otvoreni:
                    if sljedbenik == cvor[1]:
                        if cvor[3] < n[3] + udaljenost:
                            insert = False
                            break
                        else:
                            del otvoreni[otvoreni.index(cvor)]
            if insert:
                nova_vrijednost = n[3] + udaljenost + heuristika[sljedbenik]
                # Priroriteni red sortira po prvom elementu tuple-a (heuristika sljedebnika + njegova udaljenost od
                # pocetnog stanja), ako je ista ta vrijednost onda po drugom elementu (stanje) odnono abecedno
                heapq.heappush(otvoreni, (nova_vrijednost, sljedbenik, n[1], n[3] + udaljenost))
    return False, None, None


# Funkcija formatiranog ispisa za algoritme UCS i A*
def ispis_astar_ufc(broj_posjecenih, rjesenje, visited, pocetno_stanje, alg):
    print("[FOUND_SOLUTION]: yes")
    print(f"[STATES_VISITED]: {broj_posjecenih}")  # Broj posjecenih stanja od pocetnog do ciljnog stanja
    roditelj = rjesenje[2]  # Roditelj ciljnog stanja
    put = [rjesenje[1]]  # Zapocinjemo put od ciljnog stanja
    # Konstruiranje puta
    while roditelj:
        put.insert(0, roditelj)
        roditelj_tuple = visited.get(roditelj)
        roditelj = roditelj_tuple[0]
        if roditelj == pocetno_stanje:
            break
    print(f"[PATH_LENGTH]: {len(put)}")  # Duljina puta
    if alg == "ucs":
        print(f"[TOTAL_COST]: {rjesenje[0]}")  # Udaljenost kod algoritma UCS
    elif alg == "astar":
        print(f"[TOTAL_COST]: {rjesenje[3]}")  # Udaljenost kod algoritma A*
    put_ispis = " => ".join(put)
    print(f"[PATH]: {put_ispis}")


def main():
    # Parsiranje linije izvršavanja
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss', type=str, required=True)
    parser.add_argument('--alg', type=str, choices=['bfs', 'ucs', 'astar'])
    parser.add_argument('--h', type=str)
    parser.add_argument('--check-optimistic', action='store_true')
    parser.add_argument('--check-consistent', action='store_true')
    args = parser.parse_args()

    # Citanje datoteke opisnika stanja - uvijek mora biti zadana
    with open(args.ss, "r", encoding="utf-8") as f:
        opisnik_stanja = f.readlines()

    # Definiranje rjecnika svih prijelaza i moguce heuristike
    opisnik_stanja = [stanje for stanje in opisnik_stanja if not stanje.startswith("#")]
    ciljna_stanja = opisnik_stanja[1].split(" ")
    ciljna_stanja = [stanje.strip() for stanje in ciljna_stanja]
    svi_prijelazi = {}
    heuristika = {}

    # Inicijalizacija rjecnika opisnika stanja
    for i in range(2, len(opisnik_stanja)):
        opisnik_stanja_parsiran = opisnik_stanja[i].split(": ")
        if len(opisnik_stanja_parsiran) > 1:
            stanje = opisnik_stanja_parsiran[0]
        else:
            stanje = opisnik_stanja_parsiran[0].rstrip(":\n")
        if len(opisnik_stanja_parsiran) > 1 and opisnik_stanja_parsiran[1]:
            sljedbenici = opisnik_stanja_parsiran[1].split(' ')
        else:
            sljedbenici = []

        pojedinacni_prijelazi = {}

        for sljedbenik in sljedbenici:
            sljedeci, udaljenost = sljedbenik.split(",")
            pojedinacni_prijelazi[sljedeci] = float(udaljenost.strip())

        svi_prijelazi[stanje.strip()] = pojedinacni_prijelazi

    # Inicijalizacija rjecnika heuristike ako je ona zadana
    if args.h:
        with open(args.h, "r", encoding="utf-8") as f:
            heuristika_lines = f.readlines()
        for line in heuristika_lines:
            stanje, heuristika_vrijednost = line.split(": ")
            heuristika[stanje] = float(heuristika_vrijednost.strip())

    # Poziv funkcije odgovarajuceg algoritma ako je on zadan
    if args.alg == "bfs":
        # Definiranje pocetnog stanja oblika tuple-a (dubina, stanje, roditelj, udaljenost od roditelja)
        pocetno_stanje = (1, opisnik_stanja[0].strip(), '', 0)
        rjesenje, broj_posjecenih, visited = bfs(pocetno_stanje, svi_prijelazi, ciljna_stanja)
        # Ispis rjesenja u zadanom formatiranom obliku kako je i zadano
        # Rjesenje kao i svako ostalo stanje je u obliku pocetnog stanja
        # ako ne postoji onda se vraca False
        if rjesenje:
            print("# BFS")
            print("[FOUND_SOLUTION]: yes")
            print(f"[STATES_VISITED]: {broj_posjecenih}")  # Broj posjecenih stanja
            print(f"[PATH_LENGTH]: {rjesenje[0]}")  # Dubina na kojoj se nalazi rjesenje
            roditelj = rjesenje[2]  # Roditelj ciljnog stanja
            put = [rjesenje[1]]  # Put zapocinjemo ciljnim stanjem
            udaljenost = rjesenje[3]  # Racunanje zapocinjemo udaljenosti ciljnog stanja od njegovog roditelja
            # Konstruiranje puta te racunanje ukupne udaljenosti puta
            while roditelj:
                put.insert(0, roditelj)
                roditelj_tuple = visited.get(roditelj)
                udaljenost += roditelj_tuple[2]  # Postupno dodavanjem udaljenosti stanja od roditelja do poc. stanja
                roditelj = roditelj_tuple[0]
                if roditelj == pocetno_stanje:
                    break
            print(f"[TOTAL_COST]: {float(udaljenost)}")
            put_ispis = " => ".join(put)
            print(f"[PATH]: {put_ispis}")
        else:
            # Ispis da rjesenje nije pronadeno ako ono uistinu ne postoji
            print("[FOUND_SOLUTION]: no")
    elif args.alg == "ucs":
        # Definiranje pocetnog stanja oblika tuple-a (udaljenost od pocetnog stanja, stanje, roditelj)
        pocetno_stanje = (0, opisnik_stanja[0].strip(), '')
        rjesenje, broj_posjecenih, visited = ucs(pocetno_stanje, svi_prijelazi, ciljna_stanja)
        # Rjesenje kao i svako ostalo stanje je u obliku pocetnog stanja
        # ako ne postoji onda se vraca False
        if rjesenje:
            print("# UCS")
            # Poziv funkcije definiranog formatiranog ispisa
            ispis_astar_ufc(broj_posjecenih, rjesenje, visited, pocetno_stanje, args.alg)
        else:
            # Ispis da rjesenje nije pronadeno ako ono uistinu ne postoji
            print("[FOUND_SOLUTION]: no")
    elif args.alg == "astar":
        # Definiranje pocetnog stanja oblika tuple-a (heuristika, stanje, roditelj, udaljenost od pocetnog stanja)
        pocetno_stanje = (0, opisnik_stanja[0].strip(), '', 0)
        rjesenje, broj_posjecenih, visited = astar(pocetno_stanje, svi_prijelazi, ciljna_stanja, heuristika)
        # Rjesenje kao i svako ostalo stanje je u obliku pocetnog stanja
        if rjesenje:
            print(f"# A-STAR {args.h}")
            # Poziv funkcije definiranog formatiranog ispisa
            ispis_astar_ufc(broj_posjecenih, rjesenje, visited, pocetno_stanje, args.alg)
        else:
            # Ispis da rjesenje nije pronadeno ako ono uistinu ne postoji
            print("[FOUND_SOLUTION]: no")

    # Ispitivanje je li heuristika optimisticna ako je zadana ta provjera
    if args.check_optimistic:
        zakljucak = "optimistic"
        print(f"# HEURISTIC-OPTIMISTIC {args.h}")
        for stanje, heuristika_vrijednost in heuristika.items():
            if stanje not in ciljna_stanja:
                pocetno_stanje = (0, stanje, '', 0)
                # Izracun najkrace udaljenosti od zadanog stanja do ciljnog stanja
                rjesenje, _, visited = ucs(pocetno_stanje, svi_prijelazi, ciljna_stanja)
                prava_udaljenost = rjesenje[0]
            else:
                prava_udaljenost = 0.0
            # Usporedivanje je li vrijednost heuristike do ciljnog stanja zapravo manja od prava udaljenosti
            # To je ujedno i uvjet optimisticnosti, ako za jedan par se to ne zadovoljava heuristika nije optimisticna
            # h(stanje) <= h*(stanje)
            if heuristika_vrijednost <= prava_udaljenost:
                presuda = "[OK]"
            else:
                presuda = "[ERR]"
                # Ako se ne zadovoljava uvjet donosi se pripadajuci zakljucak
                zakljucak = "not optimistic"
            print(f"[CONDITION]: {presuda} h({stanje}) <= h*: {heuristika_vrijednost} <= {prava_udaljenost}")
        print(f"[CONCLUSION]: Heuristic is {zakljucak}.")

    # Ispitivanje je li heuristika konzistentna ako je zadana ta provjera
    if args.check_consistent:
        zakljucak = "consistent"
        print(f"# HEURISTIC-CONSISTENT {args.h}")
        for stanje1, prijelazi in svi_prijelazi.items():
            # Inicijalizacija heuristike za stanje_1
            h_s1 = heuristika[stanje1]
            for stanje2, cijena in prijelazi.items():
                # Inicijalizacija heuristike za stanje_2 u koje se moze doci iz stanje_1
                h_s2 = heuristika[stanje2]
                # Provjera je li heuristika stanje_1 manja od heuristike stanje_2 + cijena puta iz stanje_1 u stanje_2
                # To je ujedno i uvjet konzistente heuristike
                # h(stanje_1) <= h(stanje_2) + cost(stanje_1, stanje_2)
                if h_s1 <= h_s2 + cijena:
                    presuda = "[OK]"
                else:
                    presuda = "[ERR]"
                    # Ako se ne zadovoljava uvjet donosi se pripadajuci zakljucak
                    zakljucak = "not consistent"
                print(f"[CONDITION]: {presuda} h({stanje1}) <= h({stanje2}) + c: {h_s1} <= {h_s2} + {cijena}")
        print(f"[CONCLUSION]: Heuristic is {zakljucak}.")

    return 0


if __name__ == "__main__":
    main()



PODSUMOWANIE:
1. W MLP-02 inicjalizacja wag była słaba.

Wartości aktywacji wstępnych (przed tanh) były bardzo duże bądź bardzo małe a tanh jest funkcją zgniatającą, dlatego większość aktywacji jest równa 1 bądź -1.
Powoduje to to, że po pierwsze powoduje to że te działania sprowadzają się praktycznie do niczego ponieważ prawie zawsze z funkcji aktywacji wynik jest 1 lub -1, a po drugie co ważniejsze powoduje to znikające gradienty,
ponieważ wzór na gradient lokalny dla tanh jest równy 1-t^2 gdzie t to powoduje ze gradient się zeruje i wszystkie poprzedzające parametry już się nie zmienią a przez to zdecydowanie zwiększa to ilość potrzebnych iteracji do osiągnięcia zadowalającego lossu.

ROZWIĄZANIE?

chcemy znormalizować tablice wag przy inicjalizacji, żeby to nie powodowało pusych gradientów.
Stosujemy inicjalizacje kaiminga/hea czyli randomowe W1 mnożymyprzez 1/(fan-in) gdzie fan-in to jest pierwiastek wejśc, w naszym wypadku N_embd*nof_letters i dodatkowo mnożymy to razy 5/3 dla tanh

PS. to już nie jest takie kluczowe, ponieważ teraz się stosuję **batch normalization**

2. Jako, że nadal wynik preaktywacji nie jest idealna (gaussowski) trzeba zastosować **batch normalization**

CZYM JEST BATCH NORMALIZATON?
to jest poprostu normalizacja tej preaktywacji ! czyli hpreact=

u - średnia z batcha

o - wariancja (czyli poprostu uśredniona suma kwardratów różnic wartości od średniej )

odchylenie standardowe - wariancja ^ 2

e - stała która zapobiega dzieleniu przez 0

$znormalizowany_x=(x-u)/(sqrt(o^2 +e))$

ALE, my nie CHECMY żeby zawsze były one idealnie znormalizowane. Chcemy żeby sieć nauczyła się czy lepiej jest mieć dane znormalizowane czy może lepiej się uczy mając dane mniej znormalizowane

więc dodajemy 2 paremtry do naszej sieci:

bngain - jako tablica 1 dla każdego neuronu

bnbias - jako tablica 0 dla każdego neuronu (obu parametrów sieć będzie się uczyła)

i teraz nasze hpreact=hpreact*bngain + bnbias (gdzie hpreact jest juz znormalizowane)

Przez to że my zmieniamy nasze parametry na podstawie mini-batchy to pojawia się regularyzacja która zapobiega overfittingowi, to wprowadza taki "jitter" w uczeniu. 


niestety batch normalization powoduje dużo bugów więc teraz raczej się uzywa layer normalization lub group normalization
ale bez batch normalization i innych tego typu regulyzatorow musimy bardzo dokładnie sprawdzać czy gradienty nie rozjerzadją się za bardzo ani nie zwężają. chcemy ładnego łuku :)


wykresy na które warto spojrzeć ( sprawdzić czy nie ma żadnych anomalii w naszej sieći)
-Histogramy Aktywacji (forward pass) - procent nasycenia czyli 
-Histogramy Gradientów (Backward Pass)
- Update-to-Data Ratio (Stosunek aktualizacji do wartości wag)
To jest prawdopodobnie najważniejszy wykres do debugowania Learning Rate (LR). Jeśli stosunek jest za niski (np. $10^{-5}$) -> sieć uczy się wieki (zbyt ostrożnie).Jeśli stosunek jest za wysoki (np. $10^{-1}$ lub więcej) -> sieć "skacze" i jest niestabilna.Złoty środek: Karpathy celuje w okolice $10^{-3}$ (czyli na wykresie logarytmicznym wartość -3).
- Standard Deviation of Activations (Odchylenie standardowe aktywacji warstwa po warstwie) Wykres pokazujący, czy sygnał "gaśnie" czy "wybucha" w miarę przechodzenia przez kolejne warstwy sieci.O co chodzi: Patrzymy na x.std() na wyjściu każdej warstwy.Dlaczego to ważne:Idealnie chcemy, aby odchylenie standardowe było stabilne przez całą sieć (dlatego używamy Gain $5/3$ dla Tanh).Jeśli std spada w kolejnych warstwach -> problem zanikającego gradientu.Jeśli std rośnie -> problem eksplodującego gradientu.Wpływ Batch Norm: Jeśli masz dobrze zaimplementowany Batch Norm, ten wykres będzie dość nudny (płaski), co jest dobrą wiadomością – BN wymusza, żeby std było stabilne. Ale bez BN (lub przy złej inicjalizacji) zobaczyłbyś drastyczny spadek/wzrost.
Notatki nieuporządkowane
**Problemy z 02-MLP - Słaba inicjalizacja :(
1. na początku inicjalizacji loss wynosi około 27.9 a powinien w teori wynosić 3.296 ( bo to jest log(-1/27.0)), spowodowane jest to tym że logits po pierwszej iteracji są bardzo rózne od -30 do 30
to spowalnia nauke sieci bo pierwsze przebiegi pętli są daleko od wyniku
2. problem z h(gdzie h to jest wynik  h = torch.tanh(emb.view(-1, N_EMBD * NOF_LETTERS) @ W1 + b1)), rysując histogram widzimy że h jest głównie równe 1 lub -1 a to jest bardzo złe, na bazie micrograd które pisałem,
backward dla funkcji tanh to grad+=(1-t**2)*grad gdzie t to jest wynik funkcji tanh wiec widzimy ze jezeli większośc
wyników to jest 1 lub -1 to gradient się nie będzie wgl zmieniał.
o funkcji tanh można pomyśleć jako o zaworze. Gdy jest backprop gradient leci przez funkcje az dochodzi do tanh i przez to ze tam jest tak że ten wynik jest bliski zeru to dla następnych parametrów W1,b1,C gradient się zerowy, i się nie zmieniają parametry

jak naprawić problem nr.1 ?

a) b2 nie moze byc randn tylko trzeba *0

b) W2 zmniejszamy np W2*=0.01, żeby wyniki były bliżej siebie

jak naprawić problem nr.2?
on wynika z tego ze wynik mnozenia tablicy embedingow @ W1 +b1 są za duże wieć trzeba zmniejszyć początkowe W1 i b1 :) jak? to jak w1 =randn(n,k) to dzielimy przez /n**0.5, lepiej można skorzystać z funkcji nn.init.kaiming_normal i wtedy mnożymy razy gain/sqrt(fanmode) 
gain dla tannh = 5/3

NOWE OPTYMALIZACJE:
1. Batch normalization:
checmy żeby wejście do tanh nie było ani bardzo duże ani bardzo małe chemy żeby było w miarę gaussowskie, więc na bazie pracy naukowej odkyrto że trzeba poporstu to wejście przekształcić na rozkład Gaussa.
Normalizujemy h_preactivation więc hpreact=hpreac-hpreac.mean
na marginesie czym jest odchylenie standardowe?
określa ono jak mocno dane są rozproszone od ich średniej, czym większe odchylenie tym większe rozproszenie (jest to pierwiastek z wariancji)
    odpowiadają za dostosowanie odchylenia i sredniej)

Problem z Barch normalization:
Przez nią nasza sieć uczy sie na batchach które mają swoją średnią i odchylenie i to jest fajne do treningu bo dodaje noise itd. ale problem pojawia się kiedy chcemy zrobić sprawdzić działanie naszej sieci nie mając tych batchu tylko na pojedynczych przykłądach

Rozwiązanie:
po treningu wyliczamy srednia i odchulenie na całej bazie treningowej i zamiast podczas ewaluacji dynamicznie obliczac to hpreact mamy fixed dane
**
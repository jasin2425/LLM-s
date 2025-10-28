Problemy z 02-MLP - Słaba inicjalizacja :(
1. na początku inicjalizacji loss wynosi około 27.9 a powinien w teori wynosić 3.296 ( bo to jest log(-1/27.0)), spowodowane jest to tym że logits po pierwszej iteracji są bardzo rózne od -30 do 30
to spowalnia nauke sieci bo pierwsze przebiegi pętli są daleko od wyniku
2. problem z h, rysując histogram widzimy że h jest głównie równe 1 lub -1 a to jest bardzo złe, na bazie micrograd które pisałem,
backward dla funkcji tanh to grad+=(1-t**2)*grad gdzie t to jest wynik funkcji tanh wiec widzimy ze jezeli większośc
wyników to jest 1 lub -1 to gradient się nie będzie wgl zmieniał.
o funkcji tanh można pomyśleć jako o zaworze. Gdy jest backprop gradient leci przez funkcje az dochodzi do tanh i przez to ze tam jest tak że ten wynik jest bliski zeru to dla następnych parametrów W1,b1,C gradient się zerowy, i się nie zmieniają parametry

jak naprawić problem nr.1 ?

a) b2 nie moze byc randn tylko trzeba *0

b) W2 zmniejszamy np W2*=0.01

jak naprawić problem nr.2?
on wynika z tego ze wynik mnozenia tablicy embedingow @ W1 +b1 są za duże wieć trzeba zmniejszyć początkowe W1 i b1 :) jak? to jak w1 =randn(n,k) to dzielimy przez /n**0.5, lepiej można skorzystać z funkcji nn.init.kaiming_normal i wtedy mnożymy razy gain/sqrt(fanmode) 
gain dla tannh = 5/3

NOWE OPTYMALIZACJE:
1. Batch normalization:
checmy żeby wejście do tanh było gaussowskie, więc na bazie pracy naukowej odkyrto że trzeba poporstu to wejście gaussowskie 
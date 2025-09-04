W 01-bigrams stworzyłem 2 programy które starają się utworzyć nowe imiona na bazie zbioru 32033 imion (z pliku names.txt)



1. countung\_version.py
   W tej wersji słowa tworzone są na bazie zliczania częstości występowania wszystkich par liter, następnie utworzona jest tablica prawdopodobieństw występowania danej pary i z tego losujemy imiona przy pomocy rozkładu wielomianowego. Na końcu liczymy jaki jest średni log(koszt) przez który oceniamy skuteczność modelu

   5 generated names:\['cexze', 'momasurailezitynn', 'konimittain', 'llayn', 'ka']
   negative log loss: 559951.56250, avg negative log loss: 2.45436

2. simple\_nn.py
   wersja gdzie generowane są słowa przy pomocy prostej sieci neruonowej któa otrzymuje litere i na output daje prawdopodobieństwo na kazda nastepna litere, z tego przy pomocy rozkładu wielomianowego losujemy nowe słowa. Dla kosztu 2.46 wygenerowane zostały te same słowa co w sposobie z tablicą częstości wystąpień.
   5 generated names:\['cexze', 'momasurailezitynn', 'konimittain', 'llayn', 'ka']
3. 

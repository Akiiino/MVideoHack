# MVideoHack


###### Решение тестового задания на хакатон MvideoHack


#### Смотреть в файл Solution.ipynb






Для решения используем *нейронные сети* --- свёрточные и рекуррентные --- так как они обычно показывают себя лучше в задаче обработки естественного языка. 

Обе архитектуры показывают достаточно хорошие результаты, но в силу того, что обучатся они всё же достаточно по-разному и на по-разному предобработанных данных, вместе они показывают результат лучше каждой в отдельности. 

Сами архитектуры не очень сложные, но мы используем множественные входы для того чтобы лучше учитывать разницу между тем, что обычно пишут в трёх различных полях. 

Итоговая активация --- sigmoid --- принимает значения в диапазоне (0, 1), так что предсказания сети переводятся в реальные как (x * 4) + 1.


Мы также экспериментировали с использованием более традиционных методов типа градиентного бустинга, но он давал результаты заметно хуже полученных из нейросетей.

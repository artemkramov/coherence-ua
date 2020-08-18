# Python package to evaluate the coherence of Ukrainian-language texts

This package represents a pre-trained Transformer-based coherence estimation model for a Ukrainian corpus. This model uses a neural network that was previously trained on the set of Ukrainian news. The detailed description of the model will be considered on the 2020 IEEE 2nd International Conference on Advanced Trends in Information Theory (ATIT).

## Installation
Use `pip` tool to install

`pip install coherence-ua`

Caution: package has several dependencies. Package `udpipe` requires some extra utilities to compile some parts of code.

## Usage
```
from coherence_ua.transformer_coherence import CoherenceModel

model = CoherenceModel()
text = """Мед – густа солодка маса, яку бджоли виробляють з нектару квітів.

Загалом у світі є до 320 видів меду. Вони різняться за своїми смаковими якостями та поживною цінністю.

В меді дійсно є такі поживні речовини як цинк, калій та залізо. Проте, на жаль, в дуже мізерних кількостях.

Наприклад, в одній столовій ложці меду заліза всього 0,5%. Проте цей продукт має велику кількість вуглеводів та калорій. 1 столова ложка меду еквівалентна 17 грамам цукру та 64 кілокалоріям.

Мед містить незначну кількість антиоксидантів та має доволі сильну бактеріальну дію. Антиоксиданти захищають клітини нашого організму від вільних радикалів.

Вільні радикали – це молекули, які виробляються, коли наш організм перетравлює їжу або ви були під впливом тютюну чи радіації."""

# Show output probabilities for each clique of a text (clique_number = 3) 
print(model.get_prediction_series(text))

# Evaluate the coherence of a text as the product of output probabilities
print(model.evaluate_coherence_as_product(text))

# Calculate the coherence of the text as the ratio of a number of coherent cliques over all cliques
# according to the corresponding threshold
print(model.evaluate_coherence_using_threshold(text, 0.1))

```
See folder `examples` for more details. As it can be seen from the example, model implements 3 methods:

- `get_prediction_series` - estimate the output probabilities for each clique of a text (clique_number = 3). A term "clique" implies the set of sentences of a text with an unitary offset. For instance, `<s1, s2, s3>`, `<s2, s3, s4>`, `<s3, s4, s5>` where `<si>` denotes a separate sentence.
- `evaluate_coherence_as_product` -  evaluate the coherence of a text as the product of output probabilities of cliques.
- `evaluate_coherence_using_threshold` - calculate the coherence of the text as the ratio of a number of coherent cliques over all cliques according to the given threshold.
 
=====================================================
=====================================================

# Програмний пакет Python для оцінки когерентності україномовних текстів 

Цей пакет реалізує попередньо натреновану модель оцінки когерентності україномовного корпусу на основі архітектури Transformer. Модель використовує нейронну мережу, що була натренована на множині українських новин. Детальний опис моделі буде розглянуто на конференції the 2020 IEEE 2nd International Conference on Advanced Trends in Information Theory (ATIT).

## Встановлення
Використовуйте інструмент `pip` для встановлення

`pip install coherence-ua`

Попередження: пакет містить декілька залежностей. Пакет `udpipe` потребує додаткових ресурсів для компіляції певних частин коду.

## Використання
```
from coherence_ua.transformer_coherence import CoherenceModel

model = CoherenceModel()
text = """Мед – густа солодка маса, яку бджоли виробляють з нектару квітів.

Загалом у світі є до 320 видів меду. Вони різняться за своїми смаковими якостями та поживною цінністю.

В меді дійсно є такі поживні речовини як цинк, калій та залізо. Проте, на жаль, в дуже мізерних кількостях.

Наприклад, в одній столовій ложці меду заліза всього 0,5%. Проте цей продукт має велику кількість вуглеводів та калорій. 1 столова ложка меду еквівалентна 17 грамам цукру та 64 кілокалоріям.

Мед містить незначну кількість антиоксидантів та має доволі сильну бактеріальну дію. Антиоксиданти захищають клітини нашого організму від вільних радикалів.

Вільні радикали – це молекули, які виробляються, коли наш організм перетравлює їжу або ви були під впливом тютюну чи радіації."""

# Show output probabilities for each clique of a text (clique_number = 3) 
print(model.get_prediction_series(text))

# Evaluate the coherence of a text as the product of output probabilities
print(model.evaluate_coherence_as_product(text))

# Calculate the coherence of the text as the ratio of a number of coherent cliques over all cliques
# according to the corresponding threshold
print(model.evaluate_coherence_using_threshold(text, 0.1))

```
Дивіться папку `examples` для уточнення деталей використання. Модель реалізує 3 методи:

- `get_prediction_series` - оцінка вихідних ймовірностей для кожної групи тексту (clique_number = 3). Під терміном "група" мається на увазі набір речень тексту з одинарним зсувом. Наприклад, `<s1, s2, s3>`, `<s2, s3, s4>`, `<s3, s4, s5>`, де `<si>` відповідає окремому реченню тексту.
- `evaluate_coherence_as_product` -  оцінка когерентності тексту як добутку вихідних ймовірностей груп.
- `evaluate_coherence_using_threshold` - розрахунок когерентності тексту як відношення кількості когерентних груп до їх загальної кількості відповідно до встановленого порогового значення.
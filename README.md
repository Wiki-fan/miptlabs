## Miptlabs
[![Build Status](https://travis-ci.org/Wiki-fan/miptlabs.svg?branch=master)](https://travis-ci.org/Wiki-fan/miptlabs)
[![Maintainability](https://api.codeclimate.com/v1/badges/f536ea351ea6eac694ac/maintainability)](https://codeclimate.com/github/Wiki-fan/miptlabs/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/f536ea351ea6eac694ac/test_coverage)](https://codeclimate.com/github/Wiki-fan/miptlabs/test_coverage)

Библиотека для Python, которая умеет в лабы.

Надстройка над Sympy, Numpy и Matplotlib. С Pandas тоже неплохо работает.

### Использование
`import miptlabs as ml`

Главный класс — `ml.PQ`, представляющий физическую величину.

Хранит значение `.val`, абсолютную погрешность `.sigma`, относительную погрешность `.epsilon`, размерность `.dim` (`sympy.physics.units`).
Арифметические действия автоматически пересчитывают погрешность. Печатается через `print`.

`ml.PQ` можно хранить в `np.array`, всё будет работать хорошо. Рекомендуется, впрочем, использовать `ml.pqarray` (подкласс `np.array`) со всякими удобностями и вкусностями (конвертация в `np.array` из `float`'ов, например).

`OLS` считает коэффициенты наилучшей прямой по МНК и ошибку приближения.

Графики строятся при помощи `ml.plt_pq`. Передавать ей надо обязательно `ml.pqarray`. Можно указать `ols=True`, и сразу будет нарисована аппроксимирующая прямая (по МНК). Можно отдельно рисовать её функцией `plot_OLS`.

Замечание: все эти функции только строят графики. Поэтому можно снаружи делать `plt.figure`, `plt.show()`, другие `plt.plot()` и всё, что заблагорассудится.

Вообще, лучше глянуть на примеры.

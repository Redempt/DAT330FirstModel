import pandas
import scipy
import numpy
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pyplot

pop = pandas.read_csv("data/population.csv", thousands=",")
pop.drop("CTC", axis=1, inplace=True)

pop = pop.melt(id_vars=["NAME"], var_name="year", value_name="population")
pop.rename(columns={"NAME": "town"}, inplace=True)
pop["year"] = pop["year"].astype(int)
tax = pandas.read_csv("data/tax.csv")

tax = tax.merge(pop, how="inner", on=["year", "town"])
tax["type"] = tax["type"].apply(lambda elem: 0 if elem == "Meals and Rooms" else 1)
tax = tax.groupby(["year", "town", "type", "population"], as_index=False)[["gross", "past_gross"]].sum()

# One hot encoding (max score: 0.98) winner!
# I think this helped because it allows the model to handle the towns more independently, rather than having a single
# coefficient that applies to all of the
tax = pandas.concat([tax, pandas.get_dummies(tax["town"])], axis=1)
tax.drop("town", inplace=True, axis=1)

# Ordinal encoding (max score: 0.92)
# towns = tax["town"].unique()
# towns_map = {towns[i]: i for i in range(len(towns))}
# tax["town"] = tax["town"].apply(lambda town: towns_map[town])

cols = []
ignore = {"gross"}
last_score = 0
count = 0

all_scores = {}

for col in tax.columns:
    if col in ignore:
        continue
    print(f"Calculating column {count} / {len(tax.columns)}")
    count += 1
    cols.append(col)
    x = numpy.array(tax[cols])
    if len(cols) == 1:
        x = x.reshape(-1, 1)
    y = numpy.array(tax["gross"]).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    score = model.score(x, y)
    all_scores[",".join(cols.copy())] = score
    if score < last_score:
        cols.pop()
    else:
        last_score = score

print(cols)
print(last_score)

pyplot.plot(*zip(*sorted(all_scores.items())))
pyplot.axis("off")
pyplot.show()
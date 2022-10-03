import pandas
import scipy
import numpy
import sklearn
import matplotlib.pyplot as pyplot

pop = pandas.read_csv("data/population.csv", thousands=",")
pop.drop("CTC", axis=1, inplace=True)

pop = pop.melt(id_vars=["NAME"], var_name="year", value_name="population")
pop.rename(columns={"NAME": "town"}, inplace=True)
pop["year"] = pop["year"].astype(int)
tax = pandas.read_csv("data/tax.csv")

retained = ["type", "year", "month_num", "town", "gross", "past_gross", "population"]

tax = tax.merge(pop, how="inner", on=["year", "town"])
tax = tax.filter(retained)
tax["type"] = tax["type"].apply(lambda elem: 0 if elem == "Meals and Rooms" else 1)
tax = tax.groupby(["year", "town", "type", "population"], as_index=False)[["gross", "past_gross"]].sum()

towns = tax["town"].unique()
towns_map = {towns[i]: i for i in range(len(towns))}
tax["town"].apply(lambda town: towns_map[town])

tax = pandas.concat([tax, pandas.get_dummies(tax["town"])], axis=1)
tax.drop("town", inplace=True, axis=1)
print(tax)
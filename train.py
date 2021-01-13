from LinearRegression import LinearRegression
import pandas as pd

if __name__ == "__main__":
    #Import CSV with pandas
    data = pd.read_csv("data.csv")
    #Get X and Y
    x = data["km"].values
    y = data["price"].values

    lr = LinearRegression()

    lr.train(x, y)
    print(lr.theta)
    lr.save_model()
    lr.plot_dataset(x, y)
    lr.display_cost_function()
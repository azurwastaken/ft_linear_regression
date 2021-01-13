from LinearRegression import LinearRegression

if __name__ == "__main__":
    kilometers = input("Please enter a Mileage : ") 
    if not (kilometers.isdigit()) :
        print("Error : Mileage must be a positive Integer")
        exit()
    lr = LinearRegression()
    lr.get_model()
    predicted = lr.predict(int(kilometers))
    if predicted < 0 :
        print("How can you expect this car to still work ?!")
    else :
        print("A car that as driven {} kilometers should cost approximately {} euro".format( kilometers, round(predicted,2)))
from src.experiments import Experiments

if __name__ == "__main__":
    print("Starting Evolutionary Computing Project - FrozenLake-v1 GA Optimization\n")
    exp = Experiments('config.json')
    exp.run_all()
    print("\nAll experiments completed. Check the 'results' folder for plots and tables.")

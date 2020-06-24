import neural_network
import deploy
import data_methods

def main():
    
    #
    path_filename = "../data/dataset.csv"
    
    data_methods.set_new_data(path_filename)
    
    neural_network.train_neural_network()
    
    # Define Problem
    J = 10e9 # Current Density (A/m^2)
    dt = 1e-12 # Temporal step (s)
    run_time = 0.5e-8 # run time (s)
    
    deploy.run_problem(J, dt, run_time)
    
    return 0



# Main define
if __name__ == '__main__':
    
    a = main()

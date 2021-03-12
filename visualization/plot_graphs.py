import matplotlib.pyplot as plt

def plot_iterations_vs_probsize(data):
    for key in data.keys():
        # key is problem name
        if key is not 'input_sizes':
            hill_iterations = data[key]['hill_climb']['iterations']
            genetic_iterations = data[key]['genetic']['iterations']
            mimic_iterations = data[key]['mimic']['iterations']
            sim_ann_iterations = data[key]['sim_ann']['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for {key}")
            ax.plot(
                prob_size, hill_iterations, marker='o',
                label="hill", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, genetic_iterations, marker='o',
                label="genetic", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mimic_iterations, marker='o',
                label="mimic", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, sim_ann_iterations, marker='o',
                label="sim ann", drawstyle="steps-post"
            )
            file_path = f'figures/{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            # plt.show()

def plot_time_vs_probsize(data):
    pass


def plot_hill_restarts(data):
    for key in data.keys():
        # key is problem name
        if key is not 'options':
            restart_5 = data[key][0]['iterations']
            restart_10 = data[key][10]['iterations']
            restart_20 = data[key][20]['iterations']
            restart_40 = data[key][40]['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for Hill Climb with restarts={key}")
            ax.plot(
                prob_size, restart_5, marker='o',
                label="0", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, restart_10, marker='o',
                label="10", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, restart_20, marker='o',
                label="20", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, restart_40, marker='o',
                label="40", drawstyle="steps-post"
            )
            file_path = f'figures/hill-{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            # plt.show()

def plot_sim_ann_schedules(data):
    for key in data.keys():
        # key is problem name
        if key is not 'options' and key is not 'input_sizes':
            geom = data[key]['geometric']['iterations']
            arith = data[key]['arithmetic']['iterations']
            exp = data[key]['exponential']['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for {key} Sim Ann Varying Decay Schedules")
            ax.plot(
                prob_size, geom, marker='o',
                label="geometric", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, arith, marker='o',
                label="arithmetic", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, exp, marker='o',
                label="exponential", drawstyle="steps-post"
            )
            file_path = f'figures/sim-ann-{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            plt.show()

def plot_genetic_mutation_prob(data):
    for key in data.keys():
        # key is problem name
        if key is not 'options' and key is not 'input_sizes':
            mut_1 = data[key][0.1]['iterations']
            mut_3 = data[key][0.3]['iterations']
            mut_5 = data[key][0.5]['iterations']
            mut_9 = data[key][0.9]['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for {key} Genetic Varying Mutation Prob")
            ax.plot(
                prob_size, mut_1, marker='o',
                label="10%", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_3, marker='o',
                label="30%", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_5, marker='o',
                label="50%", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_5, marker='o',
                label="90%", drawstyle="steps-post"
            )
            file_path = f'figures/genetic-{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            plt.show()

def plot_genetic_pop_size(data):
    for key in data.keys():
        # key is problem name
        if key is not 'options' and key is not 'input_sizes':
            mut_1 = data[key][50]['iterations']
            mut_3 = data[key][200]['iterations']
            mut_5 = data[key][300]['iterations']
            mut_9 = data[key][400]['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for {key} Genetic Varying Population Sizes")
            ax.plot(
                prob_size, mut_1, marker='o',
                label="50", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_3, marker='o',
                label="200", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_5, marker='o',
                label="300", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_5, marker='o',
                label="400", drawstyle="steps-post"
            )
            file_path = f'figures/genetic-popsize-{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            plt.show()

def plot_mimic_keep_pct(data):
    for key in data.keys():
        # key is problem name
        if key is not 'options' and key is not 'input_sizes':
            mut_1 = data[key][0.1]['iterations']
            mut_3 = data[key][0.5]['iterations']
            mut_5 = data[key][0.9]['iterations']
            prob_size = data['input_sizes']

            fig, ax = plt.subplots()
            ax.set_xlabel("problem size")
            ax.set_ylabel("num of iterations")
            ax.set_title(f"Iterations vs problem size for {key} MIMIC Varying Keep Pct")
            ax.plot(
                prob_size, mut_1, marker='o',
                label="10%", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_3, marker='o',
                label="50%", drawstyle="steps-post"
            )
            ax.plot(
                prob_size, mut_5, marker='o',
                label="90%", drawstyle="steps-post"
            )
            file_path = f'figures/mimic-keeppct-{key}-iter-probsize.png'
            ax.legend()
            print('saving figure....')
            fig.savefig(file_path)
            plt.show()
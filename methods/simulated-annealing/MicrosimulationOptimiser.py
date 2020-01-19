from simanneal import Annealer
import numpy as np

class MicrosimulationOptimiser(Annealer):
    """A class for performing simulated annealing on the microsimulation problem"""

    def __init__(self, inds, *cons, logs=False, pop_size=None):

        self.logs = logs

        # Check that the constraints are consistent - must contain same number of individuals
        for i, con in enumerate(cons):
            if i == 0:
                population_count = sum(con)
            else:
                assert sum(con) == population_count

        if pop_size is None:
            self.n_synth_individs = population_count
        else:
            self.n_synth_individs = pop_size

        self.individs = inds  # Use term individs rather than inds to avoid confusion with plural of id!
        self.cons = cons

        # Give each synthetic individual a random ID from the seed individual dataset, then set up the optimiser class
        state = np.random.choice(range(self.individs.shape[0]), self.n_synth_individs)
        super(MicrosimulationOptimiser, self).__init__(state)

    def move(self):

        # Select individual to be changed
        index_change = np.random.randint(self.n_synth_individs)

        if self.logs:
            print("Changing individual {}; was {}: \t {}".format(index_change, self.state[index_change], self.state))

        # Give this individual a new id, which must be different from its current id
        new_id = np.random.choice(self.individs.shape[0] - 1)
        if new_id < self.state[index_change]:
            self.state[index_change] = new_id
        else:
            self.state[index_change] = new_id + 1

        if self.logs:
            print("now {}: \t\t\t\t {}".format(self.state[index_change], self.state))

    def energy(self):

        # Generate the synthetic population: state tells us which individuals are present by their index in inds
        synthetic_individs = self.individs[self.state,]

        total_abs_error = 0

        # Count up how many individuals there are with each property
        # and compare to the counts in the constraint tables
        for s_individs, con in zip(synthetic_individs.transpose(), self.cons):

            # Count how many simulated people have each option for this constraint
            con_id, con_id_counts = np.unique(s_individs, return_counts=True)

            # Some options might be missing - add their label and frequency (0) to the list manually
            missing = np.where(np.isin(range(con.shape[0]), con_id, invert=True))
            if missing[0].size != 0:

                if self.logs:
                    print("Constraint IDs {} are missing in constraint table {}".format(missing[0], con_id))
                    for i, c in zip(con_id, con_id_counts):
                        print("[{}, {}]".format(i, c))

                con_id = np.append(con_id, missing[0])
                con_id_counts = np.append(con_id_counts, np.zeros_like(missing[0]))

                if self.logs:
                    print("Added missing constraint IDs:")
                    for i, c in zip(con_id, con_id_counts):
                        print("[{}, {}]".format(i, c))

                # Sort the resulting counts array
                con_id_counts = con_id_counts[np.argsort(con_id)]

                if self.logs:
                    con_id = con_id[np.argsort(con_id)]
                    print("Sorted constraints:")
                    for i, c in zip(con_id, con_id_counts):
                        print("[{}, {}]".format(i, c))

            abs_error = np.sum(np.abs(con - con_id_counts))
            total_abs_error += abs_error

            if self.logs:
                print("Constraint counts: {}".format(con))
                print("Synthetic counts: {}".format(con_id_counts))
                print("Error: {}".format(abs_error))

        return total_abs_error
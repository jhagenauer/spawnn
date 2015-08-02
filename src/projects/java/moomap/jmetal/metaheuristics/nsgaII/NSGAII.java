//  NSGAII.java
//
//  Author:
//       Antonio J. Nebro <antonio@lcc.uma.es>
//       Juan J. Durillo <durillo@lcc.uma.es>
//
//  Copyright (c) 2011 Antonio J. Nebro, Juan J. Durillo
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package moomap.jmetal.metaheuristics.nsgaII;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Distance;
import jmetal.util.JMException;
import jmetal.util.Ranking;
import jmetal.util.comparators.CrowdingComparator;

/**
 * This class implements the NSGA-II algorithm.
 */
public class NSGAII extends Algorithm {

	private static final long serialVersionUID = 8414491980622964783L;

	// subclass for internal usage
	class CreateOffspring implements Callable<Solution[]> {
		Problem prob;
		SolutionSet pop;
		Operator select, recomb, mutate;

		CreateOffspring(Problem prob, SolutionSet pop, Operator select, Operator recomb, Operator mutate) {
			this.prob = prob;
			this.pop = pop;
			this.select = select;
			this.recomb = recomb;
			this.mutate = mutate;
		}

		@Override
		public Solution[] call() throws Exception {
			Solution[] parents = new Solution[2];
			parents[0] = (Solution) select.execute(pop);
			parents[1] = (Solution) select.execute(pop);

			Solution[] offSpring = (Solution[]) recomb.execute(parents);
			mutate.execute(offSpring[0]);
			mutate.execute(offSpring[1]);
			problem_.evaluate(offSpring[0]);
			problem_.evaluateConstraints(offSpring[0]);
			problem_.evaluate(offSpring[1]);
			problem_.evaluateConstraints(offSpring[1]);

			return offSpring;
		}
	}

	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public NSGAII(Problem problem) {
		super(problem);
	} // NSGAII

	/**
	 * Runs the NSGA-II algorithm.
	 * 
	 * @return a <code>SolutionSet</code> that is a set of non dominated
	 *         solutions as a result of the algorithm execution
	 * @throws JMException
	 */
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		Distance distance = new Distance();

		// Read the parameters
		int threads = ((Integer) getInputParameter("threads")).intValue();
		int populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		int maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		QualityIndicator indicators = (QualityIndicator) getInputParameter("indicators");

		// Initialize the variables
		SolutionSet population = new SolutionSet(populationSize);
		int evaluations = 0;

		int requiredEvaluations = 0;

		// Read the operators
		Operator mutationOperator = operators_.get("mutation");
		Operator crossoverOperator = operators_.get("crossover");
		Operator selectionOperator = operators_.get("selection");

		// Create the initial solutionSet
		for (int i = 0; i < populationSize; i++) {
			Solution newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			population.add(newSolution);
		} // for

		// Generations
		int gen = 0;
		while (evaluations < maxEvaluations) {
			gen++;
			if ((evaluations % 10) == 0) {
				StringBuffer sb = new StringBuffer();
				sb.append(gen + "," + evaluations + ",");
				for (int i = 0; i < problem_.getNumberOfObjectives(); i++)
					sb.append(population.get(0).getObjective(i) + ",");
				System.out.println(sb.toString());
			}

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<Solution[]>> futures = new ArrayList<Future<Solution[]>>();
			for (int i = 0; i < (populationSize / 2) && evaluations < maxEvaluations; i++) {
				futures.add(es.submit(new CreateOffspring(problem_, population, selectionOperator, crossoverOperator, mutationOperator)));
				evaluations += 2;
			}
			es.shutdown();

			SolutionSet offspringPopulation = new SolutionSet(populationSize);
			for (Future<Solution[]> s : futures) {
				try {
					offspringPopulation.add(s.get()[0]);
					offspringPopulation.add(s.get()[1]);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}

			// Create the solutionSet union of solutionSet and offSpring
			SolutionSet union = ((SolutionSet) population).union(offspringPopulation);

			// Ranking the union
			Ranking ranking = new Ranking(union);

			int remain = populationSize;
			int index = 0;
			SolutionSet front = null;
			population.clear();

			// Obtain the next front
			front = ranking.getSubfront(index);

			while ((remain > 0) && (remain >= front.size())) {
				// Assign crowding distance to individuals
				distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
				// Add the individuals of this front
				for (int k = 0; k < front.size(); k++) {
					population.add(front.get(k));
				} // for

				// Decrement remain
				remain = remain - front.size();

				// Obtain the next front
				index++;
				if (remain > 0) {
					front = ranking.getSubfront(index);
				} // if
			} // while

			// Remain is less than front(index).size, insert only the best one
			if (remain > 0) { // front contains individuals to insert
				distance.crowdingDistanceAssignment(front, problem_.getNumberOfObjectives());
				front.sort(new CrowdingComparator());
				for (int k = 0; k < remain; k++) {
					population.add(front.get(k));
				} // for

				remain = 0;
			} // if

			// This piece of code shows how to use the indicator object into the
			// code
			// of NSGA-II. In particular, it finds the number of evaluations
			// required
			// by the algorithm to obtain a Pareto front with a hypervolume
			// higher
			// than the hypervolume of the true Pareto front.
			if ((indicators != null) && (requiredEvaluations == 0)) {
				double HV = indicators.getHypervolume(population);
				if (HV >= (0.98 * indicators.getTrueParetoFrontHypervolume())) {
					requiredEvaluations = evaluations;
				} // if
			} // if
		} // while

		// Return as output parameter the required evaluations
		setOutputParameter("evaluations", requiredEvaluations);

		// Return the first non-dominated front
		Ranking ranking = new Ranking(population);
		return ranking.getSubfront(0);
	} // execute
} // NSGA-II

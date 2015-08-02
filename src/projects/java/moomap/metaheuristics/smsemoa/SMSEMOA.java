//  DENSEA_main.java
//
//  Author:
//       Simon Wessing
//
//  Copyright (c) 2011 Simon Wessing
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

package moomap.metaheuristics.smsemoa;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.qualityIndicator.Hypervolume;
import jmetal.qualityIndicator.util.MetricsUtil;
import jmetal.util.JMException;
import jmetal.util.Ranking;
import jmetal.util.comparators.CrowdingDistanceComparator;

import moomap.jmetal.encodings.variable.TopoMap;

import org.apache.log4j.Logger;

import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

/**
 * This class implements the SMS-EMOA algorithm, as described in
 * 
 * Michael Emmerich, Nicola Beume, and Boris Naujoks. An EMO algorithm using the
 * hypervolume measure as selection criterion. In C. A. Coello Coello et al.,
 * Eds., Proc. Evolutionary Multi-Criterion Optimization, 3rd Int'l Conf. (EMO
 * 2005), LNCS 3410, pp. 62-76. Springer, Berlin, 2005.
 * 
 * and
 * 
 * Boris Naujoks, Nicola Beume, and Michael Emmerich. Multi-objective
 * optimisation using S-metric selection: Application to three-dimensional
 * solution spaces. In B. McKay et al., Eds., Proc. of the 2005 Congress on
 * Evolutionary Computation (CEC 2005), Edinburgh, Band 2, pp. 1282-1289. IEEE
 * Press, Piscataway NJ, 2005.
 */

public class SMSEMOA extends Algorithm {

	private static Logger log = Logger.getLogger(SMSEMOA.class);

	private static final long serialVersionUID = -2977110156595909943L;
	
	/**
	 * stores the problem to solve
	 */
	private MetricsUtil utils_;
	private Hypervolume hv_;

	/**
	 * Constructor
	 * 
	 * @param problem
	 *            Problem to solve
	 */
	public SMSEMOA(Problem problem) {
		super(problem);
		this.utils_ = new jmetal.qualityIndicator.util.MetricsUtil();
		this.hv_ = new Hypervolume();
	} // SMSEMOA

	/**
	 * Runs the SMS-EMOA algorithm.
	 * 
	 * @return a <code>SolutionSet</code> that is a set of non dominated
	 *         solutions as a result of the algorithm execution
	 * @throws JMException
	 */
	public SolutionSet execute() throws JMException, ClassNotFoundException {
		// Read the parameters
		int populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		int maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		double offset = (Double) getInputParameter("offset");

		// Initialize the variables
		SolutionSet population = new SolutionSet(populationSize);
		int evaluations = 0;
		
		// Read the operators
		Operator mutationOperator = operators_.get("mutation");
		Operator crossoverOperator = operators_.get("crossover");
		Operator selectionOperator = operators_.get("selection");

		// Create the initial solutionSet
		Solution newSolution;
		for (int i = 0; i < populationSize; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			evaluations++;
			population.add(newSolution);
		} // for

		// Generations ...
		while (evaluations < maxEvaluations) {
			
			if( evaluations % 1000 == 0 ) {
				Hypervolume hv = new Hypervolume();
				
				Ranking ranking = new Ranking(population);
				SolutionSet ss = ranking.getSubfront(0);
				
				int numObjs = problem_.getNumberOfObjectives();
				double[][] front = ss.writeObjectivesToMatrix();							
				double d1 = hv.calculateHypervolume(front, ss.size(), numObjs );
				
				log.info("Evals: "+evaluations+", Hv: "+d1 );
			}
			
			// Only for eu-opt
			if( evaluations % 500000 == 0 ) {			
				Ranking ranking = new Ranking(population);
				SolutionSet ss = ranking.getSubfront(0);
								
				try {
					File dir = new File("output/"+evaluations+"/");
			    	dir.mkdirs();
			    	
					List<double[]> objs = new ArrayList<double[]>();
				    for( int i = 0; i < population.size(); i++ ) {
				    	Solution s = ss.get(i);
				    	
				    	double[] d = new double[problem_.getNumberOfObjectives()];
				    	for( int j = 0; j < problem_.getNumberOfObjectives(); j++ )
				    		d[j] = s.getObjective(j);
				    	objs.add(d);
				    	
				    	TopoMap tm = (TopoMap)s.getDecisionVariables()[0];
				    	SomUtils.saveGrid(tm.grid_, new FileOutputStream( dir.getAbsolutePath()+"/grid"+i+".xml"));
				    }
				    DataUtils.writeCSV( new FileOutputStream( dir.getAbsolutePath()+"/objectives.csv"), objs, new String[]{"fe", "ftopo", "ge", "gtopo" },';' );
				    log.info("Saved.");
				} catch( Exception e ) {
					e.printStackTrace();
				}	
			}
						
			// select parents
			LinkedList<Solution> selectedParents = new LinkedList<Solution>();
			Solution[] parents = new Solution[0];
			while (selectedParents.size() < 2) {
				Object selected = selectionOperator.execute(population);
				try {
					Solution parent = (Solution) selected;
					selectedParents.add(parent);
				} catch (ClassCastException e) {
					parents = (Solution[]) selected;
					for (Solution parent : parents) {
						selectedParents.add(parent);
					}
				}
			}
			parents = selectedParents.toArray(parents);

			// crossover
			Solution[] offSpring = (Solution[]) crossoverOperator.execute(parents);

			// mutation
			mutationOperator.setParameter("evaluation", evaluations);
			mutationOperator.execute(offSpring[0]);

			// evaluation
			problem_.evaluate(offSpring[0]);
			problem_.evaluateConstraints(offSpring[0]);

			// insert child into the offspring population
			SolutionSet offspringPopulation = new SolutionSet(populationSize);
			offspringPopulation.add(offSpring[0]);

			evaluations++;

			// Create the solutionSet union of solutionSet and offSpring
			SolutionSet union = ((SolutionSet) population).union(offspringPopulation);

			// Ranking the union (non-dominated sorting)
			Ranking ranking = new Ranking(union);

			// ensure crowding distance values are up to date
			// (may be important for parent selection)
			for (int j = 0; j < population.size(); j++) {
				population.get(j).setCrowdingDistance(0.0);
			}

			SolutionSet lastFront = ranking.getSubfront(ranking.getNumberOfSubfronts() - 1);
			if (lastFront.size() > 1) {
				double[][] frontValues = lastFront.writeObjectivesToMatrix();
				int numberOfObjectives = problem_.getNumberOfObjectives();
				// STEP 1. Obtain the maximum and minimum values of the Pareto
				// front
				double[] maximumValues = utils_.getMaximumValues(union.writeObjectivesToMatrix(), numberOfObjectives);
				double[] minimumValues = utils_.getMinimumValues(union.writeObjectivesToMatrix(), numberOfObjectives);
				// STEP 2. Get the normalized front
				double[][] normalizedFront = utils_.getNormalizedFront(frontValues, maximumValues, minimumValues);
				// compute offsets for reference point in normalized space
				double[] offsets = new double[maximumValues.length];
				for (int i = 0; i < maximumValues.length; i++) {
					offsets[i] = offset / (maximumValues[i] - minimumValues[i]);
				}
				// STEP 3. Inverse the pareto front. This is needed because the
				// original
				// metric by Zitzler is for maximization problems
				double[][] invertedFront = utils_.invertedFront(normalizedFront);
				// shift away from origin, so that boundary points also get a
				// contribution > 0
				for (double[] point : invertedFront) {
					for (int i = 0; i < point.length; i++) {
						point[i] += offsets[i];
					}
				}

				// calculate contributions and sort
				double[] contributions = hvContributions(invertedFront);
				for (int i = 0; i < contributions.length; i++) {
					// contribution values are used analogously to crowding
					// distance
					lastFront.get(i).setCrowdingDistance(contributions[i]);
				}

				lastFront.sort(new CrowdingDistanceComparator());
			}

			// all but the worst are carried over to the survivor population
			SolutionSet front = null;
			population.clear();
			for (int i = 0; i < ranking.getNumberOfSubfronts() - 1; i++) {
				front = ranking.getSubfront(i);
				for (int j = 0; j < front.size(); j++) {
					population.add(front.get(j));
				}
			}
			for (int i = 0; i < lastFront.size() - 1; i++) {
				population.add(lastFront.get(i));
			}

		} // while

		// Return the first non-dominated front
		Ranking ranking = new Ranking(population);
		return ranking.getSubfront(0);
	} // execute

	/**
	 * Calculates how much hypervolume each point dominates exclusively. The
	 * points have to be transformed beforehand, to accommodate the assumptions
	 * of Zitzler's hypervolume code.
	 * 
	 * @param front
	 *            transformed objective values
	 * @return HV contributions
	 */
	private double[] hvContributions(double[][] front) {
		int numberOfObjectives = problem_.getNumberOfObjectives();
		double[] contributions = new double[front.length];
		double[][] frontSubset = new double[front.length - 1][front[0].length];
		LinkedList<double[]> frontCopy = new LinkedList<double[]>();
		for (double[] point : front) {
			frontCopy.add(point);
		}
		double[][] totalFront = frontCopy.toArray(frontSubset);
		double totalVolume = hv_.calculateHypervolume(totalFront, totalFront.length, numberOfObjectives);
		for (int i = 0; i < front.length; i++) {
			double[] evaluatedPoint = frontCopy.remove(i);
			frontSubset = frontCopy.toArray(frontSubset);
			// STEP4. The hypervolume (control is passed to java version of
			// Zitzler code)
			double hv = hv_.calculateHypervolume(frontSubset, frontSubset.length, numberOfObjectives);
			double contribution = totalVolume - hv;
			contributions[i] = contribution;
			// put point back
			frontCopy.add(i, evaluatedPoint);
		}
		return contributions;
	}
} // SMSEMOA

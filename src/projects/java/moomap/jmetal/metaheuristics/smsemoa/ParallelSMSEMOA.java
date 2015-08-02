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

package moomap.jmetal.metaheuristics.smsemoa;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

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

public class ParallelSMSEMOA extends Algorithm {

	private static Logger log = Logger.getLogger(ParallelSMSEMOA.class);

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
	public ParallelSMSEMOA(Problem problem) {
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
		final int populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		final int maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		final double offset = (Double) getInputParameter("offset");
		int threads = ((Integer) getInputParameter("threads")).intValue();
		final boolean localOpt = (Boolean)getInputParameter("localOpt");
					
		// Initialize the variables
		final SolutionSet population = new SolutionSet(populationSize);
		
		class EvalCounter {
			private int c = 0;
			public int get() { return c; }
			public void inc(){ c = c + 1; }
		}
				
		final EvalCounter evals = new EvalCounter();
						
		// Read the operators
		final Operator mutationOperator = operators_.get("mutation");
		final Operator crossoverOperator = operators_.get("crossover");
		final Operator selectionOperator = operators_.get("selection");
		
		final Lock lock = new ReentrantLock();
				
		// Create the initial solutionSet
		Solution newSolution;
		for (int i = 0; i < populationSize; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			
			if( localOpt )
				localOpt( newSolution, mutationOperator );
			
			population.add(newSolution);
			evals.inc();
		} // for
		
		// Generations ...
		System.out.println("threads: "+threads);
		ExecutorService es = new NotifyingBlockingThreadPoolExecutor(threads, threads*2, 60, TimeUnit.MINUTES);
				
		while (evals.get() < maxEvaluations) {		
			es.submit( new Runnable() {
				@Override
				public void run() {
					lock.lock();	
					// select parents
					LinkedList<Solution> selectedParents = new LinkedList<Solution>();
					Solution[] parents = new Solution[0];
					try {
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
					} catch( JMException e ) {
						e.printStackTrace();
					}
					lock.unlock();

					Solution[] offSpring = null;
					try {
						// crossover
						offSpring = (Solution[]) crossoverOperator.execute(parents);

						// mutation
						mutationOperator.execute(offSpring[0]);
						problem_.evaluate(offSpring[0]);
						problem_.evaluateConstraints(offSpring[0]);
										
						
					} catch( JMException e ) {
						e.printStackTrace();
					}
					if( localOpt )
						offSpring[0] = localOpt( offSpring[0], mutationOperator );
					
					lock.lock();
					// insert child into the offspring population
					SolutionSet offspringPopulation = new SolutionSet(populationSize);
					offspringPopulation.add(offSpring[0]);

					// Create the solutionSet union of solutionSet and offSpring
					SolutionSet union = ((SolutionSet)population).union(offspringPopulation);

					// Ranking the union (non-dominated sorting)
					Ranking ranking = new Ranking(union);

					// ensure crowding distance values are up to date (may be important for parent selection)
					for (int j = 0; j < population.size(); j++) 
						population.get(j).setCrowdingDistance(0.0);
					
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
						// STEP 3. Inverse the pareto front. This is needed because the original metric by Zitzler is for maximization problems
						double[][] invertedFront = utils_.invertedFront(normalizedFront);
						// shift away from origin, so that boundary points also get a contribution > 0
						for (double[] point : invertedFront) 
							for (int i = 0; i < point.length; i++) 
								point[i] += offsets[i];
												
						// calculate contributions and sort
						double[] contributions = hvContributions(invertedFront);
						for (int i = 0; i < contributions.length; i++) // contribution values are used analogously to crowding distance
							lastFront.get(i).setCrowdingDistance(contributions[i]);
						
						lastFront.sort(new CrowdingDistanceComparator());
					}

					// all but the worst are carried over to the survivor population
					population.clear();
					for (int i = 0; i < ranking.getNumberOfSubfronts() - 1; i++) {
						SolutionSet front = ranking.getSubfront(i);
						for (int j = 0; j < front.size(); j++) 
							population.add(front.get(j));
					}
					for (int i = 0; i < lastFront.size() - 1; i++) 
						population.add(lastFront.get(i));
										
					// output to console/file
					if( evals.get() % 100 == 0 ) {
						Hypervolume hv = new Hypervolume();
						SolutionSet ss = new Ranking(population).getSubfront(0);
						double[][] front = ss.writeObjectivesToMatrix();
						double d1 = hv.calculateHypervolume( front, ss.size(), problem_.getNumberOfObjectives() );
						
						int numObjs = problem_.getNumberOfObjectives();
						double[] maximumValues = utils_.getMaximumValues(population.writeObjectivesToMatrix(), numObjs);
						double[] minimumValues = utils_.getMinimumValues(population.writeObjectivesToMatrix(), numObjs);
						double[][] normalizedFront = utils_.getNormalizedFront(ss.writeObjectivesToMatrix(), maximumValues, minimumValues);		
										
						double d2 = hv.calculateHypervolume(normalizedFront, ss.size(), numObjs );
						
						double[][] invertedFront = utils_.invertedFront(normalizedFront);	
						double d3 = hv.calculateHypervolume(invertedFront, ss.size(), numObjs );
						
						log.info(evals.get()+", "+d1+", "+d2+", "+d3 );
					}
								
					if( evals.get() % 10000 == 0 ) {
						log.debug(evals.get()+" evals, writing...");
						
						// local opt
						/*SolutionSet ss = new SolutionSet(populationSize);
						for( int i = 0; i < population.size(); i++ ) 
							localOpt(population.get(i), mutationOperator );
						
						ss = new Ranking(ss).getSubfront(0);
						
						Hypervolume hv = new Hypervolume();
						double[][] front = ss.writeObjectivesToMatrix();
						double d1 = hv.calculateHypervolume( front, ss.size(), problem_.getNumberOfObjectives() );
						log.debug("opt hv: "+d1);*/
						
						SolutionSet ss = new Ranking(population).getSubfront(0);
															
						try {
							File dir = new File("output/"+evals.get()+"/");
					    	dir.mkdirs();
					    	
							List<double[]> objs = new ArrayList<double[]>();
						    for( int i = 0; i < ss.size(); i++ ) { // population? no, we want the front!
						    	Solution s = ss.get(i);
						    	
						    	double[] d = new double[problem_.getNumberOfObjectives()];
						    	for( int j = 0; j < problem_.getNumberOfObjectives(); j++ )
						    		d[j] = s.getObjective(j);
						    	objs.add(d);
						    	
						    	TopoMap tm = (TopoMap)s.getDecisionVariables()[0];
						    	SomUtils.saveGrid(tm.grid_, new FileOutputStream( dir.getAbsolutePath()+"/grid"+i+".xml"));
						    }
						    DataUtils.writeCSV( new FileOutputStream( dir.getAbsolutePath()+"/objectives.csv"), objs, new String[]{"fe", "ftopo", "ge", "gtopo" },';' );
						} catch( Exception e ) {
							e.printStackTrace();
						}	
						log.debug("done.");
					}
																				
					evals.inc(); // bad style but works
					lock.unlock();
					
				} // while
			});
			
			
		}
		es.shutdown();
		
		// Return the first non-dominated front
		Ranking ranking = new Ranking(population);
		return ranking.getSubfront(0);
	} // execute
	
	public Solution localOpt( Solution s, Operator op ) {
		Solution best = s;
		try {
			for( int j = 0; j < 50; j++ ) {
				Solution cur = new Solution(best);
				op.execute(cur);
				problem_.evaluate(cur);
				problem_.evaluateConstraints(cur);
				
				boolean curDominates = true;
				for( int k = 0; k < problem_.getNumberOfObjectives(); k++ )
					if( cur.getObjective(k) > best.getObjective(k) )
						curDominates = false;
				
				if( curDominates ) {
					best = cur;
					j=0;
				}
			}
		}catch( JMException e ) {
			e.printStackTrace();
		}
		return best;
	}
	
	/**
	 * Calculates how much hypervolume each point dominates exclusively. The
	 * points have to be transformed beforehand, to accommodate the assumptions
	 * of Zitzler's hypervolume code.
	 * 
	 * @param front transformed objective values
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

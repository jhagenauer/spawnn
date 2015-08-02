//  ssGAS.java
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

package moomap.jmetal.metaheuristics.singleObjective.geneticAlgorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.operators.selection.WorstSolutionSelection;
import jmetal.util.JMException;
import jmetal.util.comparators.ObjectiveComparator;
import jmetal.util.wrapper.XReal;

import moomap.jmetal.metaheuristics.smsemoa.NotifyingBlockingThreadPoolExecutor;

import org.apache.log4j.Logger;

import spawnn.utils.DataUtils;

public class ParallelSSGA extends Algorithm {
	
	private static final long serialVersionUID = -6772964058521080538L;
	private static Logger log = Logger.getLogger(ParallelSSGA.class);

	public ParallelSSGA(Problem problem) {
		super(problem);
	} // SSGA

	public SolutionSet execute() throws JMException, ClassNotFoundException {
				
		class EvalCounter {
			private int c = 0;
			public int get() { return c; }
			public void inc(){ c = c + 1; }
		}
		final EvalCounter evals = new EvalCounter();
		
		int threads = ((Integer) getInputParameter("threads")).intValue();
		final Lock lock = new ReentrantLock();

		final Comparator comparator = new ObjectiveComparator(0); 

		HashMap parameters; // Operator parameters
		parameters = new HashMap();
		parameters.put("comparator", comparator);
		final Operator findWorstSolution = new WorstSolutionSelection(parameters);

		// Read the parameters
		int populationSize = ((Integer) this.getInputParameter("populationSize")).intValue();
		int maxEvaluations = ((Integer) this.getInputParameter("maxEvaluations")).intValue();

		// Initialize the variables
		final SolutionSet population = new SolutionSet(populationSize);
		
		// Read the operators
		final Operator mutationOperator = this.operators_.get("mutation");
		final Operator crossoverOperator = this.operators_.get("crossover");
		final Operator selectionOperator = this.operators_.get("selection");

		// Create the initial population
		ExecutorService es = new NotifyingBlockingThreadPoolExecutor(threads, threads*2, 60, TimeUnit.MINUTES);
		List<Future<Solution>> futures = new ArrayList<Future<Solution>>();
				
		for (int i = 0; i < populationSize; i++) {	
			futures.add( es.submit( new Callable<Solution>() {
				@Override
				public Solution call() {	
					Solution newIndividual = null; 
					try {
						newIndividual = new Solution(problem_);
						problem_.evaluate(newIndividual);
					} catch( JMException e ) {
						e.printStackTrace();
						System.exit(1);
					} catch (ClassNotFoundException e) {
						e.printStackTrace();
						System.exit(1);
					}
					return newIndividual;
				}
			}));
		} // for
		es.shutdown();
		
		for( Future<Solution> f : futures ) {
			evals.inc();
			try {
				population.add(f.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}

		// main loop
		es = new NotifyingBlockingThreadPoolExecutor(threads, threads*2, 60, TimeUnit.MINUTES);
				
		while (evals.get() < maxEvaluations) {
			es.submit( new Runnable() {
				@Override
				public void run() {	
					try {
					
						lock.lock();
						Solution[] parents = new Solution[2];
						// Selection
						parents[0] = (Solution) selectionOperator.execute(population);
						parents[1] = (Solution) selectionOperator.execute(population);
						lock.unlock();

						Solution[] offspring = (Solution[]) crossoverOperator.execute(parents);
						mutationOperator.execute(offspring[0]);
						problem_.evaluate(offspring[0]);
						
						lock.lock();
						evals.inc();
						// Replacement: replace the last individual is the new
						// one is better
						int worstIndividual = (Integer) findWorstSolution.execute(population);
						if (comparator.compare(population.get(worstIndividual), offspring[0]) > 0) {
							population.remove(worstIndividual);
							population.add(offspring[0]);
						} // if
						
						if( evals.get() % 100 == 0 ) {
							Solution best = population.best(comparator);
							log.debug(evals.get()+" : "+best.getObjective(0) );
							
							XReal vars = new XReal(best);
							
							List<double[]> samples = new ArrayList<double[]>();
						    for (int i = 0 ; i < best.numberOfVariables(); i+=4) {
						    	
						    	double t = 0; // target class
						    	if(vars.getValue(i+3) <= 0.5 )
						    		t = 1;
						    			
						    	samples.add( new double[]{ 
						    			vars.getValue(i), // x
						    			vars.getValue(i+1), // y
						    			vars.getValue(i+2), // v
						    			t
						    		} );
						    }
						    DataUtils.writeCSV("output/"+evals.get()+".csv", samples, new String[]{"x","y","v","c"} );
						}
						lock.unlock();
					} catch( JMException e ) {
						e.printStackTrace();
						System.exit(1);
					}
				}
			});

		} // while

		// Return a population with the best individual

		SolutionSet resultPopulation = new SolutionSet(1);
		resultPopulation.add(population.best(comparator));

		System.out.println("Evaluations: " + evals.get());
		return resultPopulation;
	} // execute
} // ssGA

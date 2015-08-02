package moomap;

import java.io.IOException;
import java.util.HashMap;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.nsgaII.NSGAII;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.operators.selection.SelectionFactory;
import jmetal.problems.Schaffer;
import jmetal.util.Configuration;
import jmetal.util.JMException;

/* Multi objective optimization
 * Schedule: 
 * 1st: single objective, iris
 * 2nd: multi-objective, iris
 * 3rd: multi-objective, spatial
 */
public class MooSom {
	public static Logger logger_;
	public static FileHandler fileHandler_;

	public static void main( String[] args ) {
		try {
			
			logger_ = Configuration.logger_ ;
		    fileHandler_ = new FileHandler("moosom.log"); 
		    logger_.addHandler(fileHandler_) ;
		
			Problem problem =  new Schaffer("Real"); //new TopoMapOptimization();
			Algorithm algorithm = new NSGAII(problem);
						
			// Number of solutions depends on max avaluations... why?
			algorithm.setInputParameter("populationSize",100);
			algorithm.setInputParameter("maxEvaluations",250000);
	
		    HashMap<String, Double> parameters = new HashMap<String,Double>() ;
		    parameters.put("probability", 0.9) ;
		    parameters.put("distributionIndex", 20.0) ;
		    Operator crossover = CrossoverFactory.getCrossoverOperator("SBXCrossover", parameters);                   

		    parameters = new HashMap<String,Double>() ;
		    parameters.put("probability", 1.0/problem.getNumberOfVariables()) ;
		    parameters.put("distributionIndex", 20.0) ;
		    Operator mutation = MutationFactory.getMutationOperator("PolynomialMutation", parameters);                    

		    Operator selection = SelectionFactory.getSelectionOperator("BinaryTournament2", null) ;                           

		    algorithm.addOperator("crossover",crossover);
		    algorithm.addOperator("mutation",mutation);
		    algorithm.addOperator("selection",selection);

		    long initTime = System.currentTimeMillis();
		    SolutionSet population = algorithm.execute();
		    long estimatedTime = System.currentTimeMillis() - initTime;
		
		    logger_.info("Total execution time: "+estimatedTime + "ms");
		    logger_.info("Variables values have been writen to file VAR");
		    population.printVariablesToFile("VAR");    
		    logger_.info("Objectives values have been writen to file FUN");
		    population.printObjectivesToFile("FUN");
		  	
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (JMException e) {
			e.printStackTrace();
		}	
	}
}

package moomap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import moomap.jmetal.operators.crossover.TopoMapNBCrossover;
import moomap.jmetal.operators.mutation.RndVectorMutation;
import moomap.jmetal.problems.TopoMapSingle;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.singleObjective.geneticAlgorithm.gGA;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;

public class Exp {
	public static Logger log;
	public static FileHandler fileHandler_;

	public static void main( String[] args ) {
		try {
			
			
			
			Random r = new Random();
			List<double[]> samples = new ArrayList<double[]>();
			for( int i = 0; i < 250; i++ ) 
				samples.add( new double[]{r.nextDouble(),r.nextDouble()});
			
			final Dist<double[]> dist = new EuclideanDist();
			final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);
			
			log = Configuration.logger_ ;
		    fileHandler_ = new FileHandler("moosom.log"); 
		    log.addHandler(fileHandler_) ;
		   	   
		    double sumA = 0;
		    for( int j = 0; j < 25; j++ ) {
		
				Problem problem =  new TopoMapSingle( 5, 5, samples, bg, dist );
				Algorithm algorithm = new gGA(problem);
	
				algorithm.setInputParameter("populationSize",40);
				algorithm.setInputParameter("maxEvaluations",50000);
		
			    HashMap<String, Object> parameters = new HashMap<String,Object>() ;
			    parameters.put("probability", 0.9) ; // 0.9
			    Operator crossover = new TopoMapNBCrossover(parameters);
			    //Operator crossover = new TopoMapUniformCrossover(parameters);
	
			    parameters = new HashMap<String,Object>() ;
			    parameters.put("probability", 1.0/(6*9) ) ;
			    Operator mutation = new RndVectorMutation(parameters);                   
	
			    Operator selection = new BinaryTournament(null) ;                           
			    algorithm.addOperator("crossover",crossover);
			    algorithm.addOperator("mutation",mutation);
			    algorithm.addOperator("selection",selection);
	
			   
			    SolutionSet population = algorithm.execute();
			   
			    
			    Solution best = null;
			    for( int i = 0; i < population.size(); i++ ) {	
			    	if( best == null || population.get(i).getObjective(0) < best.getObjective(0) )
			    		best = population.get(i);
			    }
			    
			    sumA += best.getObjective(0)/25;
		    }
		    log.info("sumA: "+sumA);
		  		     		  		  	
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

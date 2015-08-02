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
import moomap.jmetal.problems.TopoMapMulti;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.nsgaII.NSGAII;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;

public class MultiObjTopoMapOpt {
	public static Logger log;
	public static FileHandler fileHandler_;

	public static void main( String[] args ) {
		try {
			int xDim = 5, yDim = 5;
			
			Random r = new Random();
			List<double[]> samples = new ArrayList<double[]>();
			for( int i = 0; i < 250; i++ ) 
				samples.add( new double[]{r.nextDouble(),r.nextDouble()});
			
			final Dist<double[]> dist = new EuclideanDist();
			final BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(dist);
			
			log = Configuration.logger_ ;
		    fileHandler_ = new FileHandler("multiopt.log"); 
		    log.addHandler(fileHandler_) ;
		
			Problem problem =  new TopoMapMulti( xDim, yDim, samples, bg, dist );
			Algorithm algorithm = new NSGAII(problem);
						
			algorithm.setInputParameter("populationSize",40);
			algorithm.setInputParameter("maxEvaluations",50000);
	
		    HashMap<String, Object> parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 0.9) ; // 0.9
		    Operator crossover = new TopoMapNBCrossover(parameters);
		    TopoMapNBCrossover.recombType = 0;
		    
		    parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 1.0/(xDim*yDim) ) ;
		    Operator mutation = new RndVectorMutation(parameters);      
		    
		    Operator selection = new BinaryTournament(null) ;                           
		    algorithm.addOperator("crossover",crossover);
		    algorithm.addOperator("mutation",mutation);
		    algorithm.addOperator("selection",selection);

		    long initTime = System.currentTimeMillis();
		    SolutionSet population = algorithm.execute();
		    long estimatedTime = System.currentTimeMillis() - initTime;
		
		    log.info("Total execution time: "+estimatedTime + "ms");

		    population.printVariablesToFile("output/VAR");    
		    population.printObjectivesToFile("output/FUN");
		    	    		  		  	
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

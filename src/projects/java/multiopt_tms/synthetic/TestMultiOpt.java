package multiopt_tms.synthetic;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.JMException;

import moomap.jmetal.encodings.variable.TopoMap;
import moomap.jmetal.metaheuristics.smsemoa.SMSEMOA;
import moomap.jmetal.operators.crossover.TopoMapNBCrossover;
import moomap.jmetal.operators.mutation.RndVectorMutation;
import moomap.jmetal.problems.SpatialTopoMapMulti;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TestMultiOpt {
	
	private static Logger log = Logger.getLogger(TestMultiOpt.class);
	
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main( String[] args ) {
		try {
			int xDim = 5, yDim = 5;
			
			List<double[]> samples = DataUtils.readCSV("data/multiopt.csv");
						
			int[] fc = new int[]{ 2 };
			int[] gc = new int[]{ 0,1 };
						
			Dist<double[]> eDist = new EuclideanDist();
			Dist<double[]> gDist = new EuclideanDist( gc );
			Dist<double[]> fDist = new EuclideanDist( fc );
			Dist<double[]> aDist = new EuclideanDist( new int[]{0,1,2} );
									
			Problem problem =  new SpatialTopoMapMulti( xDim, yDim, samples, aDist, gDist, fDist );
			//Algorithm algorithm = new NSGAII(problem);
			Algorithm algorithm = new SMSEMOA(problem);
			//Algorithm algorithm = new ParallelSMSEMOA(problem);
			algorithm.setInputParameter("offset", 100.0);
						
			algorithm.setInputParameter("populationSize",40);
			algorithm.setInputParameter("maxEvaluations", (int)(1* Math.pow(10, 4)) );
				
		    HashMap<String, Object> parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 0.9) ; // 0.9
		    Operator crossover = new TopoMapNBCrossover(parameters);
		    
		    TopoMapNBCrossover.recombType = 0;
		    		    
		    parameters = new HashMap<String,Object>() ;
		    parameters.put("probability", 1.0/(xDim*yDim) ) ; 
		    Operator mutation = new RndVectorMutation(parameters);
		    
		    Operator selection = new BinaryTournament(null);                           
		    algorithm.addOperator("crossover",crossover);
		    algorithm.addOperator("mutation",mutation);
		    algorithm.addOperator("selection",selection);

		    long initTime = System.currentTimeMillis();
		    SolutionSet population = algorithm.execute();	
		    log.info("Total execution time: "+((System.currentTimeMillis()-initTime)/1000) + "s");
		    
		    population.printVariablesToFile("output/VAR");    
		    population.printObjectivesToFile("output/FUN");
		    
		    List<double[]> objs = new ArrayList<double[]>();
		    for( int i = 0; i < population.size(); i++ ) {
		    	Solution s = population.get(i);
		    	
		    	double[] d = new double[problem.getNumberOfObjectives()];
		    	for( int j = 0; j < problem.getNumberOfObjectives(); j++ )
		    		d[j] = s.getObjective(j);
		    	objs.add(d);
		    	
		    	TopoMap tm = (TopoMap)s.getDecisionVariables()[0];
		    	SomUtils.saveGrid(tm.grid_, new FileOutputStream("output/grid"+i+".xml"));
		    	SomUtils.printUMatrix(tm.grid_, fDist, "output/");
		    	System.out.println(Arrays.toString(d)+"-> map "+i);
		    }
		    DataUtils.writeCSV( new FileOutputStream("output/objectives.csv"), objs, new String[]{"fe", "ftopo", "ge", "gtopo" },';' );
		    
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

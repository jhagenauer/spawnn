import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Set;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.metaheuristics.singleObjective.geneticAlgorithm.gGA;
import jmetal.operators.selection.BinaryTournament;
import jmetal.util.JMException;
import moomap.jmetal.encodings.variable.TopoMap;
import moomap.jmetal.operators.crossover.TopoMapNBCrossover;
import moomap.jmetal.operators.mutation.RndVectorMutation;
import moomap.jmetal.problems.TopoMapSingle;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.ColorBrewerUtil.ColorMode;

public class IrisTestGA {

	private static Logger log = Logger.getLogger(IrisTestGA.class);
	
	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 10000;
		int X_DIM = 5;
		int Y_DIM = 5;
						
		List<double[]> samples = null;
		try {
			
			samples = DataUtils.readCSV( new FileInputStream(new File("data/iris.csv") ) );			
			DataUtils.normalize(samples);
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
				
		Dist<double[]> eDist = new EuclideanDist();
		BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(eDist);
		
		Grid2D<double[]> grid = null;
		
		try {
			int xDim = 5, yDim = 5;
				
			
			Problem problem =  new TopoMapSingle(xDim, yDim, samples, bg, eDist);
			Algorithm algorithm = new gGA(problem);
			algorithm.setInputParameter("offset", 100.0);
						
			algorithm.setInputParameter("populationSize",40);
			algorithm.setInputParameter("maxEvaluations", (int)(4* Math.pow(10, 4)) );
				
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
		    
		    Solution s = population.get(0);
		    TopoMap tm = (TopoMap)s.getDecisionVariables()[0];   	
		    grid = tm.grid_;
		 
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (JMException e) {
			e.printStackTrace();
		}	
									
		log.debug("qe: "+SomUtils.getMeanQuantError( grid, bg, eDist, samples ) );
		log.debug("te: "+ SomUtils.getTopoError( grid, bg, samples ) );
		
		log.debug("spearman topo: "+SomUtils.getTopoCorrelation(samples, grid, bg, eDist, SomUtils.SPEARMAN_TYPE) );
		log.debug("pearson topo: "+SomUtils.getTopoCorrelation(samples, grid, bg, eDist, SomUtils.PEARSON_TYPE) );
		
		try {
			SomUtils.printDMatrix( grid, eDist, new FileOutputStream( "output/irisDmatrixGA.png" ) );
			SomUtils.printUMatrix( grid, eDist, new FileOutputStream( "output/irisUMatrixGA.png" ) );
						
			int[][] ws = SomUtils.getWatershed(45, 255, 2.0, grid, eDist, false);
			Collection<Set<GridPos>> clusters = SomUtils.getClusterFromWatershed(ws, grid);
			log.debug("clusters: "+clusters.size() );
			
			double[][] dImg = new double[ws.length][ws[0].length];
			for( int i = 0; i < ws.length; i++ )
				for( int j = 0; j < ws[i].length; j++ )
					dImg[i][j] = ws[i][j];
			
			SomUtils.printImage( SomUtils.getRectMatrixImage( dImg, 50, ColorMode.Greys), new FileOutputStream("output/watershed.png") );	

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}

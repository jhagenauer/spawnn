package gwr.ga;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import ga.GeneticAlgorithm;
import regioClust.LinearModel;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWR_GA_Expdat {

	private static Logger log = Logger.getLogger(GWR_GA_Expdat.class);

	public static void main(String[] args) {		
		
		Random r = new Random(0);
		GWKernel kernel = GWKernel.gaussian;
		boolean adaptive = true;
				
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/expdat/expdat.shp"), new int[]{5,6,7,9,10} ,true);
		
		List<double[]> samples = sdf.samples;
		log.debug(samples.size());
				
		int bwInit = -1;
		int[] fa = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		String[] faNames = new String[]{ "lnareatot","lnareapl","age","condh1","heat1","cellar1","acad","garage1","terr1"};
		int[] ga = new int[] { 10, 11 };
		int ta = 0;
						
		GWRIndividualCostCalculatorCorrelation cc_cor = new GWRIndividualCostCalculatorCorrelation(samples, fa, ga, ta, kernel, adaptive, faNames );	
		GWRCostCalculator cc_aic = new GWRIndividualCostCalculatorAICc(samples, fa, ga, ta, kernel, adaptive );
		GWRCostCalculator cc_cv = new GWRIndividualCostCalculatorCV(samples, fa, ga, ta, kernel, adaptive, 10 );
		GWRCostCalculator cc = cc_aic;
						
		int minGene = 2; 
		int maxGene = samples.size();
		{
			GWRIndividualAdaptive i = null;
			log.info("Search global bandwidth/j");
			double bestCost = Double.MAX_VALUE;
			for (int j = 2; j < 100; j++) {
				
				List<Integer> bw = new ArrayList<>();
				while( bw.size() < samples.size() )
					bw.add( j );
				GWRIndividualAdaptive ind = new GWRIndividualAdaptive(bw, j, Integer.MAX_VALUE);

				double cost = cc.getCost(ind);
				if (cost < bestCost) {
					bestCost = cost;
					bwInit = j;
					i = ind;
					log.debug(j+", "+cost);
				}
			}
			GWRIndividualCostCalculatorAICc.debug = true;
			log.info("best bw "+bwInit+", score: " + cc.getCost(i) );
			log.info("mean sign cor: "+cc_cor.getCost(i) );
			GWRIndividualCostCalculatorAICc.debug  = false;
		}
		
		GWRIndividualAdaptive.sd = 8.0;
				
		GeneticAlgorithm.tournamentSize = 2;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		GeneticAlgorithm.threads = 7;
		GeneticAlgorithm.maxK = 40000;
		GeneticAlgorithm.maxNoImpro = 100;

		log.debug("init...");
		List<GWRIndividualAdaptive> init = new ArrayList<GWRIndividualAdaptive>();
		
		while (init.size() < 50) {
			List<Integer> bandwidth = new ArrayList<>();
			while (bandwidth.size() < samples.size())
				bandwidth.add( bwInit + r.nextInt(17)-8 );
			GWRIndividualAdaptive i = new GWRIndividualAdaptive(bandwidth, minGene, maxGene );
			init.add( i );
		}
										
		log.debug("search (GA)...");
		
		GeneticAlgorithm<GWRIndividualAdaptive> gen = new GeneticAlgorithm<GWRIndividualAdaptive>();
		GWRIndividualAdaptive result = (GWRIndividualAdaptive) gen.search(init, cc);
		Map<double[], Double> resultBw = cc.getSpatialBandwidth(result);
		
		log.debug( "mean sign cor: "+cc_cor.getCost(result) );
		GWRIndividualCostCalculatorAICc.debug = true;
		cc_aic.getCost(result);
		GWRIndividualCostCalculatorAICc.debug = false;

		{
			DoubleMatrix Y = new DoubleMatrix(LinearModel.getY(samples, ta));
			DoubleMatrix X = new DoubleMatrix(LinearModel.getX(samples, fa, true));
			
			List<double[]> rr = new ArrayList<double[]>();
			for (int i = 0; i < samples.size(); i++) {
				double[] a = samples.get(i);
				double bw = resultBw.get(a);
				
				// build XtW
				DoubleMatrix XtW = new DoubleMatrix(X.getColumns(), X.getRows());
				for (int j = 0; j < X.getRows(); j++) {
					double[] b = samples.get(j);
					double d = new EuclideanDist(ga).dist(a, b);
					double w = GeoUtils.getKernelValue(kernel, d, bw );

					XtW.putColumn(j, X.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(X);
				
				try {
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
					double pred = X.getRow(i).mmul(beta).get(0);
					double[] c = concatenate( new double[]{ a[ga[0]], a[ga[1]], bw, result.getChromosome().get(i), pred, a[ta], pred-a[ta] }, beta.data );
					rr.add( c );
				} catch( LapackException e ) {
					//System.err.println("Couldn't solve eqs! Too low bandwidth?! "+bw+", "+adaptive+", "+resultBw.get(i) );
				}
			}		
			String s = "output/result_expdat_" + cc_cv.getCost(result)+"_"+cc_aic.getCost(result);
			String[] h = concatenate( new String[]{ "long", "lat", "radius_dist","chromosome_i", "prediction", "actual", "residual", "intercept" } , faNames );
			DataUtils.writeCSV(s + ".csv", rr, h );
		}	
	}
	
	public static double[] concat( double[] a, double[] b ) {
		double[] c = new double[a.length+b.length];
		for( int i = 0; i < a.length; i++ )
			c[i] = a[i];
		for( int i = 0; i < b.length; i++ )
			c[a.length+i] = b[i];
		return c;
	}
	
	public static <T> T concatenate(T a, T b) {
	    if (!a.getClass().isArray() || !b.getClass().isArray()) {
	        throw new IllegalArgumentException();
	    }

	    Class<?> resCompType;
	    Class<?> aCompType = a.getClass().getComponentType();
	    Class<?> bCompType = b.getClass().getComponentType();

	    if (aCompType.isAssignableFrom(bCompType)) {
	        resCompType = aCompType;
	    } else if (bCompType.isAssignableFrom(aCompType)) {
	        resCompType = bCompType;
	    } else {
	        throw new IllegalArgumentException();
	    }

	    int aLen = Array.getLength(a);
	    int bLen = Array.getLength(b);

	    @SuppressWarnings("unchecked")
	    T result = (T) Array.newInstance(resCompType, aLen + bLen);
	    System.arraycopy(a, 0, result, 0, aLen);
	    System.arraycopy(b, 0, result, aLen, bLen);        

	    return result;
	}	
}

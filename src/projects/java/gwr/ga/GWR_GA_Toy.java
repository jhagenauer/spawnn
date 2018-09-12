package gwr.ga;

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

public class GWR_GA_Toy {

	private static Logger log = Logger.getLogger(GWR_GA_Toy.class);

	public static void main(String[] args) {

		Random r = new Random(0);
		GWKernel kernel = GWKernel.boxcar;
		boolean adaptive = true;
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 1000 ) {
			double lon = r.nextDouble();
			double lat = r.nextDouble();
			
			double beta1 = 1, beta2 = 1;;
			if( lon > 0.5 )
				beta1 = -1;
			if( lat > 0.5 )
				beta2 = -1;
			double x1 = r.nextDouble();
			double x2 = r.nextDouble();
			double y = x1 * beta1 + x2 * beta2;
						
			double[] d = new double[]{ lon, lat, x1, x2, y, beta1, beta2 };
			samples.add(d);
		}
		log.debug(samples.size());
		DataUtils.writeCSV("output/spDat.csv", samples, new String[] { "long", "lat", "x1", "x2", "y", "beta1", "beta2" });
		
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 2, 3 };
		int ta = 4;
		String[] faNames = new String[]{"x1", "x2"};
		
		GWRIndividualCostCalculatorCorrelation cc_cor = new GWRIndividualCostCalculatorCorrelation(samples, fa, ga, ta, kernel, adaptive, faNames );
		GWRCostCalculator cc_aic = new GWRIndividualCostCalculatorAICc(samples, fa, ga, ta, kernel, adaptive );
		GWRCostCalculator cc_cv = new GWRIndividualCostCalculatorCV(samples, fa, ga, ta, kernel, adaptive, 10 );
		GWRCostCalculator cc = cc_cv;

		int bwInit = -1;
		int minGene = 2; 
		int maxGene = samples.size();
		{
			GWRIndividualAdaptive i = null;
			log.info("Search global bandwidth/j");
			double bestCost = Double.MAX_VALUE;
			for (int j = 3; j < 50; j++) {
				
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
			log.info("best bw "+bwInit+", score: " + cc.getCost(i) );
			log.info("cor: "+cc_cor.getCost(i) );
		}
		
		GWRIndividualAdaptive.sd = 8;
				
		GeneticAlgorithm.tournamentSize = 2;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		GeneticAlgorithm.threads = 7;
		GeneticAlgorithm.maxK = 40000;
		GeneticAlgorithm.maxNoImpro = 100;

		log.debug("init...");
		List<GWRIndividualAdaptive> init = new ArrayList<GWRIndividualAdaptive>();
	
		while (init.size() < 50 ) {
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
		
		log.info("cv: " + cc_cv.getCost(result) + ", aic: "+cc_aic.getCost(result));
		//cc_cor.getCost(result);
		
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
					double[] c = GWR_GA_Expdat.concatenate( new double[]{ a[ga[0]], a[ga[1]], bw, result.getChromosome().get(i), pred, a[ta], pred-a[ta] }, beta.data );
					rr.add( c );
				} catch( LapackException e ) {
					//System.err.println("Couldn't solve eqs! Too low bandwidth?! "+bw+", "+adaptive+", "+resultBw.get(i) );
				}
			}
			
			boolean debug = true;
			GWRIndividualCostCalculatorCV.debug = debug;
			GWRIndividualCostCalculatorAICc.debug = debug;
			
			String s = "output/result_" + cc_cv.getCost(result)+"_"+cc_aic.getCost(result);
			String[] h = GWR_GA_Expdat.concatenate( new String[]{ "long", "lat", "radius_dist","chromosome_i", "prediction", "actual", "residual", "intercept" } , faNames );
			DataUtils.writeCSV(s + ".csv", rr, h );
		}	
	}
}

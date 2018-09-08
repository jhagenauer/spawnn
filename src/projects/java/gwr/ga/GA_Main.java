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

public class GA_Main {

	private static Logger log = Logger.getLogger(GA_Main.class);

	public static void main(String[] args) {
		
		// aicc
		// boxcar no impro 20000, cost -4835.937201825724,  rate -0.24202678553754686
		// gaussia no impro 14773, cost -1634.3616558505848, rate -0.11138565091328187
		// bisquar no impro 20000, cost -4018.7868393618255, rate -0.20108009803671698
		
		// cv
		
		Random r = new Random(0);
		GWKernel kernel = GWKernel.boxcar;
		boolean adaptive = true;
		
		int bwInit = -1;
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 3 };
		int ta = 4;
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 1000 ) {
			double lon = r.nextDouble();
			double lat = r.nextDouble();
			
			double beta = 1;
			if( lon > 0.4 && lon < 0.8 && lat > 0.4 && lat < 0.8 )
			//if( lon > 0.5 )
				beta = -1;
			double x1 = r.nextDouble();
			double y = x1 + beta;
						
			double[] d = new double[]{ lon, lat, beta, x1, y};
			samples.add(d);
		}
		log.debug(samples.size());
		DataUtils.writeCSV("output/spDat.csv", samples, new String[] { "long", "lat", "beta", "x1", "y" });
		
		GWRIndividualCostCalculator_Correlation cc_cor = new GWRIndividualCostCalculator_Correlation(samples, fa, ga, ta, kernel, adaptive, new String[]{"x1"} );
		GWRCostCalculator cc_aic = new GWRIndividualCostCalculator_AICc(samples, fa, ga, ta, kernel, adaptive );
		GWRCostCalculator cc_cv = new GWRIndividualCostCalculator_CV(samples, fa, ga, ta, kernel, adaptive, 10 );
		GWRCostCalculator cc = cc_cv;
		
		/*{ // test
			List<Integer> bw = new ArrayList<>();
			while( bw.size() < samples.size() )
				bw.add( 19 );
			GWRIndividual ind = new GWRIndividual(bw, 1, Integer.MAX_VALUE);
			
			cc_aic = new GWRIndividualCostCalculator_AICc(samples, fa, ga, ta, GWKernel.boxcar, true );
			GWRIndividualCostCalculator_AICc.debug = true;
			
			cc_aic.getCost(ind);
			System.exit(1);
		}*/
				
		int minGene = 2; 
		int maxGene = samples.size();
		log.debug(minGene+" : "+maxGene);
		{
			GWRIndividual i = null;
			log.info("Search global bandwidth/j");
			double bestCost = Double.MAX_VALUE;
			for (int j = 4; j < 50; j++) {
				
				List<Integer> bw = new ArrayList<>();
				while( bw.size() < samples.size() )
					bw.add( j );
				GWRIndividual ind = new GWRIndividual(bw, j, Integer.MAX_VALUE);

				double cost = cc.getCost(ind);
				if (cost < bestCost) {
					bestCost = cost;
					bwInit = j;
					i = ind;
					log.debug(j+", "+cost);
				}
			}
			GWRIndividualCostCalculator_CV.debug = true;
			log.info("best bw "+bwInit+", " + cc.getCost(i) );
			cc_cor.getCost(i);
			GWRIndividualCostCalculator_CV.debug = false;
		}
				
		GeneticAlgorithm.tournamentSize = 3;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		GeneticAlgorithm.threads = 7;
		GeneticAlgorithm.maxK = 20000;
		GeneticAlgorithm.maxNoImpro = 100;

		log.debug("init...");
		List<GWRIndividual> init = new ArrayList<GWRIndividual>();
		
		while (init.size() < 10) {
			List<Integer> bandwidth = new ArrayList<>();
			while (bandwidth.size() < samples.size())
				bandwidth.add( bwInit + r.nextInt(9)-4 );
			GWRIndividual i = new GWRIndividual(bandwidth, minGene, maxGene );
			init.add( i );
		}
										
		log.debug("search (GA)...");
		GeneticAlgorithm<GWRIndividual> gen = new GeneticAlgorithm<GWRIndividual>();
		GWRIndividual result = (GWRIndividual) gen.search(init, cc);
		Map<double[], Double> resultBw = cc.getSpatialBandwidth(result);
		
		log.info("cv: " + cc_cv.getCost(result) + ", aic: "+cc_aic.getCost(result));
		cc_cor.getCost(result);

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
					rr.add(new double[] { a[ga[0]], a[ga[1]], beta.data[0], beta.data[1], bw, result.getChromosome().get(i), pred, a[ta], pred-a[ta] });
				} catch( LapackException e ) {
					//System.err.println("Couldn't solve eqs! Too low bandwidth?! "+bw+", "+adaptive+", "+resultBw.get(i) );
				}
			}
			
			boolean debug = true;
			GWRIndividualCostCalculator_CV.debug = debug;
			GWRIndividualCostCalculator_AICc.debug = debug;
			
			String s = "output/result_" + cc_cv.getCost(result)+"_"+cc_aic.getCost(result);
			DataUtils.writeCSV(s + ".csv", rr, new String[] { "long", "lat", "intercept", "b1","radius_dist","chromosome_i", "prediction", "actual", "residual" });
		}	
	}
}

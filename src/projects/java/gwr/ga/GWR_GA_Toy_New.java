package gwr.ga;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.exceptions.LapackException;

import heuristics.ga.GeneticAlgorithm;
import regioClust.LinearModel;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWR_GA_Toy_New {

	private static Logger log = Logger.getLogger(GWR_GA_Toy_New.class);

	public static enum mode {
		adaptive, fixed
	};

	public static void main(String[] args) {

		Random r = new Random(0);
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 1000 ) {
			double lon = r.nextDouble();
			double lat = r.nextDouble();
			
			double beta1 = 1, beta2 = 1;
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
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		GeneticAlgorithm.tournamentSize = 2;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		GeneticAlgorithm.threads = 7;
		GeneticAlgorithm.maxK = 40000;
		GeneticAlgorithm.maxNoImpro = 100;
		
		mode m = mode.adaptive;

		int initSize = 50;
		GWKernel kernel = GWKernel.boxcar;
	
		GWRIndividual<?> result2 = null;
		String fn = null;
		if (m == mode.adaptive) {
			
			GWRIndividualCostCalculatorCorrelation<GWRIndividualAdaptive> cc_cor = new GWRIndividualCostCalculatorCorrelation<GWRIndividualAdaptive>(samples, fa, ga, ta, kernel, faNames);
			GWRCostCalculator<GWRIndividualAdaptive> cc_aic = new GWRIndividualCostCalculatorAICc<GWRIndividualAdaptive>(samples, fa, ga, ta, kernel);
			GWRCostCalculator<GWRIndividualAdaptive> cc_cv = new GWRIndividualCostCalculatorCV<GWRIndividualAdaptive>(samples, fa, ga, ta, kernel, 10);
			GWRCostCalculator<GWRIndividualAdaptive> cc = cc_aic;
						
			int bwInit = -1;
			int minGene = 2;
			int maxGene = samples.size();
			{
				GWRIndividualAdaptive i = null;
				log.info("Search global bandwidth/j");
				double bestCost = Double.MAX_VALUE;
				for (int j = 4; j < 100; j++) {

					List<Integer> bw = new ArrayList<>();
					while (bw.size() < samples.size())
						bw.add(j);
					GWRIndividualAdaptive ind = new GWRIndividualAdaptive(bw, j, Integer.MAX_VALUE);

					double cost = cc.getCost(ind);
					if (cost < bestCost) {
						bestCost = cost;
						bwInit = j;
						i = ind;
						log.debug(j + ", " + cost);
					}
				}
				log.info("best bw " + bwInit + ", score: " + cc.getCost(i));
				log.info("mean sign cor: " + cc_cor.getCost(i));
			}

			GWRIndividualAdaptive.sd = 8.0;

			log.debug("init...");
			List<GWRIndividualAdaptive> init = new ArrayList<GWRIndividualAdaptive>();

			while (init.size() < initSize) {
				List<Integer> bandwidth = new ArrayList<>();
				while (bandwidth.size() < samples.size())
					bandwidth.add(bwInit + r.nextInt(17) - 8);
				GWRIndividualAdaptive i = new GWRIndividualAdaptive(bandwidth, minGene, maxGene);
				init.add(i);
			}

			log.debug("search (GA)...");

			GeneticAlgorithm<GWRIndividualAdaptive> gen = new GeneticAlgorithm<GWRIndividualAdaptive>();
			GWRIndividualAdaptive result = (GWRIndividualAdaptive) gen.search(init, cc);
			result2 = result;
			
			log.info("aic: " + cc_aic.getCost(result)+", cv: "+cc_cv.getCost(result));
			log.debug("mean sign cor: " + cc_cor.getCost(result));
			
			fn = "output/result_expdat_adaptive_" + cc_cv.getCost(result) + "_" + cc_aic.getCost(result)+".csv";
		} else {
			GWRIndividualCostCalculatorCorrelation<GWRIndividualFixed> cc_cor = new GWRIndividualCostCalculatorCorrelation<GWRIndividualFixed>(samples, fa, ga, ta, kernel, faNames);
			GWRCostCalculator<GWRIndividualFixed> cc_aic = new GWRIndividualCostCalculatorAICc<GWRIndividualFixed>(samples, fa, ga, ta, kernel);
			GWRCostCalculator<GWRIndividualFixed> cc_cv = new GWRIndividualCostCalculatorCV<GWRIndividualFixed>(samples, fa, ga, ta, kernel, 10);
			GWRCostCalculator<GWRIndividualFixed> cc = cc_cv;
			
			SummaryStatistics ss = new SummaryStatistics();
			for( int i = 0; i < samples.size()-1; i++ )
				for( int j = i+1; j < samples.size(); j++ )
					ss.addValue( gDist.dist(samples.get(i), samples.get(j)));
			
			double bwInit = -1;
			double minGene = ss.getMin();
			double maxGene = ss.getMax();
			{
				GWRIndividualFixed i = null;
				log.info("Search global bandwidth/j");
				double bestCost = Double.MAX_VALUE;
				for (double j = minGene; j <= maxGene; j+= (maxGene-minGene)/1000 ) {

					List<Double> bw = new ArrayList<>();
					while (bw.size() < samples.size())
						bw.add(j);
					GWRIndividualFixed ind = new GWRIndividualFixed(bw, j, Double.MAX_VALUE);

					double cost = cc.getCost(ind);
					if (cost < bestCost) {
						bestCost = cost;
						bwInit = j;
						i = ind;
						log.debug(j + ", " + cost);
					}
				}
				GWRIndividualCostCalculatorAICc.debug = true;
				log.info("best bw " + bwInit + ", score: " + cc.getCost(i));
				log.info("mean sign cor: " + cc_cor.getCost(i));
				GWRIndividualCostCalculatorAICc.debug = false;
			}

			GWRIndividualFixed.sd = 0.02;

			log.debug("init...");
			List<GWRIndividualFixed> init = new ArrayList<GWRIndividualFixed>();

			while (init.size() < initSize) {
				List<Double> bandwidth = new ArrayList<>();
				while (bandwidth.size() < samples.size())
					bandwidth.add(bwInit + r.nextGaussian() * GWRIndividualFixed.sd );
				GWRIndividualFixed i = new GWRIndividualFixed(bandwidth, minGene, maxGene);
				init.add(i);
			}

			log.debug("search (GA)...");

			GeneticAlgorithm<GWRIndividualFixed> gen = new GeneticAlgorithm<GWRIndividualFixed>();
			GWRIndividualFixed result = (GWRIndividualFixed) gen.search(init, cc);
			result2 = result;
			
			log.info("aic: " + cc_aic.getCost(result)+", cv: "+cc_cv.getCost(result));
			log.debug("mean sign cor: " + cc_cor.getCost(result));
			
			fn = "output/result_expdat_fixed_" + cc_cv.getCost(result) + "_" + cc_aic.getCost(result)+".csv";
		}

		{
			Map<double[], Double> resultBw = result2.getSpatialBandwidth(samples, gDist );
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
					double w = GeoUtils.getKernelValue(kernel, d, bw);

					XtW.putColumn(j, X.getRow(j).mul(w));
				}
				DoubleMatrix XtWX = XtW.mmul(X);

				try {
					DoubleMatrix beta = Solve.solve(XtWX, XtW.mmul(Y));
					double pred = X.getRow(i).mmul(beta).get(0);
					double[] c = concatenate(new double[] { a[ga[0]], a[ga[1]], bw, Double.parseDouble(result2.geneToString(i)), pred, a[ta], pred - a[ta] }, beta.data);
					rr.add(c);
				} catch (LapackException e) {}
			}
			String[] h = concatenate(new String[] { "long", "lat", "radius_dist", "chromosome_i", "prediction", "actual", "residual"}, concatenate( faNames, new String[]{"intercept"} ) );
			DataUtils.writeCSV(fn, rr, h);
		}
	}

	public static double[] concat(double[] a, double[] b) {
		double[] c = new double[a.length + b.length];
		for (int i = 0; i < a.length; i++)
			c[i] = a[i];
		for (int i = 0; i < b.length; i++)
			c[a.length + i] = b[i];
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

package llm;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.Grid2DHexToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.BubbleKernel;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;

public class TestOneLLMSOM {

	private static Logger log = Logger.getLogger(TestOneLLMSOM.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		Map<Integer, Set<double[]>> cl = new HashMap<Integer, Set<double[]>>();

		int MAX_SAMPLES = 10000;
		for( int i = 0;  i < MAX_SAMPLES; i++ ) { 
			int c;
			double[] s, d;
			s = new double[] { r.nextDouble() };	
			if (s[0] < 0.5 ) {
				d = new double[] { 10 * s[0] };
				c = 0;
			} else {
				d = new double[] { 0.1 * s[0] };
				c = 1;
			}
			
			samples.add(s);
			desired.add(d);
			if (!cl.containsKey(c))
				cl.put(c, new HashSet<double[]>());
			cl.get(c).add(s);
		}
		final String[] names = new String[] { "x" };
		
		Dist<double[]> eDist = new EuclideanDist(); 
		final Dist<double[]> fDist = new EuclideanDist();

		Grid2D<double[]> grid = new Grid2DHex<double[]>(6,5);
		log.debug(grid.getMaxDist());
		SomUtils.initRandom(grid, samples);
		
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);
		bmuGetter = new ErrorBmuGetter(samples, desired);
		
		LLMSOM llm = new LLMSOM(
				new GaussKernel(new LinearDecay(grid.getMaxDist()/2, 0.01 )), new LinearDecay(0.5, 0.05), grid, bmuGetter, 
                new GaussKernel(new LinearDecay(grid.getMaxDist()/2, 0.01 )), new LinearDecay(1.0, 0.05),
                new int[] { 0 }, 1);
		
		((ErrorBmuGetter)bmuGetter).setLLMSOM(llm);
		
		for (int t = 0; t < T_MAX; t++) {
			int j = r.nextInt(samples.size());
			llm.train((double) t / T_MAX, samples.get(j), desired.get(j));		
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples) {
			response.add(llm.present(x));
		}
		
		Map<GridPos, Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);

		ColorMode cm = ColorMode.Blues;
		try {
			SomUtils.printHexUMat(grid, fDist, cm, new FileOutputStream("output/umatrix.png"));
			SomUtils.printDMatrix(grid, fDist, cm, new FileOutputStream("output/dmatrix.png"));
			SomUtils.printClassDist(cl.values(), mapping, grid, "output/classes.png");

			for (int i = 0; i < samples.get(0).length; i++) {
				SomUtils.printComponentPlane(grid, i, cm, new FileOutputStream("output/" + names[i] + ".png"));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		// build new grid for coeff
		{
			Grid2D<double[]> ng = new Grid2DHex<double[]>(grid.getSizeOfDim(0), grid.getSizeOfDim(1));
			for (GridPos p : grid.getPositions())
				ng.setPrototypeAt(p, llm.matrix.get(p)[0]);

			try {
				SomUtils.printDMatrix(ng, eDist, cm, new FileOutputStream("output/dmatrix_coef.png"));

				for (int i = 0; i < samples.get(0).length; i++) {
					SomUtils.printComponentPlane(ng, i, cm, new FileOutputStream("output/" + names[i] + "_coef.png"));
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}

			//log.debug("------ Coeff: --------");
		}

		// build new grid for intercept
		{
			Grid2D<double[]> ng = new Grid2DHex<double[]>(grid.getSizeOfDim(0), grid.getSizeOfDim(1));
			for (GridPos p : grid.getPositions())
				ng.setPrototypeAt(p, llm.output.get(p));
			try {
				SomUtils.printDMatrix(ng, eDist, cm, new FileOutputStream("output/dmatrix_inter.png"));
				
				for (int i = 0; i < llm.output.values().iterator().next().length; i++) {
					SomUtils.printComponentPlane(ng, i, cm, new FileOutputStream("output/" + names[i] + "_inter.png"));
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}		
		
		double sdCoeff = 0;
		double sdInter = 0;
		int c = 0;
		// ------- print stats ------- 
		for (GridPos p : grid.getPositions()) {
			Set<double[]> s = mapping.get(p);
			double[] w = grid.getPrototypeAt(p);
			if (s.isEmpty())
				continue;
			
			log.debug("----------------------");
			log.debug(p + "," + s.size());
			log.debug("w: " + Arrays.toString(w));
			{
				{ // LLM
					List<double[]> des = new ArrayList<double[]>();
					List<double[]> resp = new ArrayList<double[]>();
					for( double[] d : s ) {
						resp.add( llm.present(d));
						des.add( desired.get(samples.indexOf(d)));
					}
					log.debug("LLM o: " + Arrays.toString( llm.output.get(p))+ ", m:" + Arrays.toString(llm.matrix.get(p)[0]) );
					log.debug("RMSE: "+Meuse.getRMSE(resp, des)+", R2: "+Math.pow(Meuse.getPearson(resp, des),2.0));
					
					int i = 0;
					for( double[] d : s ) 
						;//log.debug((i++)+","+d[0]+","+llm.getResponse(d, p)[0]);
					
				}
				
				{ // OLS B
					double[] y = new double[s.size()];
					double[][] x = new double[s.size()][];
					int l = 0;
					for( double[] d : s ) {
						int idx = samples.indexOf(d);
						y[l] = desired.get(idx)[0];
						x[l] = samples.get(idx);
						x[l] = Arrays.copyOf(x[l], x[l].length);
						for( int i = 0; i < x[l].length; i++ ) //subtract prototype
							x[l][i] -= w[i];
						l++;
					}
				
					OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
					ols.setNoIntercept(false);
					ols.newSampleData(y, x);
					double[] beta = ols.estimateRegressionParameters();
										
					sdCoeff += Math.pow( beta[1] - llm.matrix.get(p)[0][0], 2 );
					sdInter += Math.pow( beta[0] - llm.output.get(p)[0], 2 );
					//llm.output.put(p,new double[]{ beta[0]} );
					//llm.matrix.put(p, new double[][]{ {beta[1] } });
					
					log.debug("OLS B:"+Arrays.toString(beta) );
					
					List<double[]> des = new ArrayList<double[]>();
					List<double[]> resp = new ArrayList<double[]>();
					for (int i = 0; i < x.length; i++) {
						double[] xi  = Arrays.copyOf(x[i], x[i].length);
						double ps = beta[0]; // intercept at beta[0]
						for (int j = 1; j < beta.length; j++)
							ps += beta[j] * xi[j - 1];

						resp.add(new double[] { ps });
						des.add( new double[]{y[i]} );
					}
					log.debug("RMSE: "+Meuse.getRMSE(resp, des)+", R2: "+Math.pow(Meuse.getPearson(resp, des),2.0));
				}
			}
			c++;
		}
			
		log.debug("------ SOM: --------");
		log.debug("SDs: "+Math.sqrt(sdInter/c)+", "+Math.sqrt(sdCoeff/c));
		log.debug("QE: "+SomUtils.getQuantizationError(grid, bmuGetter, fDist, samples));
		log.debug("TE: "+SomUtils.getTopoError(grid, bmuGetter, samples));
		log.debug("RMSE: " + Meuse.getRMSE(response, desired));
		log.debug("R^2:  " + Math.pow(Meuse.getPearson(response, desired), 2));
	}
}

package llm;

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

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.ExponentialDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.som.decay.SigmoidDecay;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TestOneLLMSOM2 {

	private static Logger log = Logger.getLogger(TestOneLLMSOM2.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		Map<Integer, Set<double[]>> cl = new HashMap<Integer, Set<double[]>>();

		int MAX_SAMPLES = 1000;
		for( int i = 0;  i < MAX_SAMPLES; i++ ) { 
			int c;
			double[] s, d;
			s = new double[] { r.nextDouble() };	
			if (s[0] < 0.5 ) {
				d = new double[] { s[0] };
				c = 0;
			} else {
				d = new double[] { 10 * s[0] };
				c = 1;
			}
			
			samples.add(s);
			desired.add(d);
			if (!cl.containsKey(c))
				cl.put(c, new HashSet<double[]>());
			cl.get(c).add(s);
		}
		
		final Dist<double[]> fDist = new EuclideanDist();

		Grid2D<double[]> grid = new Grid2DHex<double[]>(6,5);
		SomUtils.initRandom(grid, samples);
		log.debug(grid.getMaxDist());
				
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);
		LLMSOM som = new LLMSOM(
				new GaussKernel(new PowerDecay(grid.getMaxDist()/2, 0.2 )), new PowerDecay(1.0, 0.005), 
				grid, bmuGetter,            
				new GaussKernel(new PowerDecay(grid.getMaxDist()/2, 0.05 )), new PowerDecay(1, 0.5),
				new int[] { 0 }, 1);
			
		for (int t = 0; t < T_MAX; t++) {
			int j = r.nextInt(samples.size());
			som.train((double) t / T_MAX, samples.get(j), desired.get(j));
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples) {
			response.add(som.present(x));
		}
				
		Map<GridPos, Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
		
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
			log.debug(p + "," + p.hashCode()+","+ s.size());
			log.debug("w: " + Arrays.toString(w));
			{
				{ // LLM
					List<double[]> des = new ArrayList<double[]>();
					List<double[]> resp = new ArrayList<double[]>();
					for( double[] d : s ) {
						resp.add( som.present(d));
						des.add( desired.get(samples.indexOf(d)));
					}
					log.debug("LLM o: " + Arrays.toString( som.output.get(p))+ ", m:" + Arrays.toString(som.matrix.get(p)[0]) );
					log.debug("RMSE: "+Meuse.getRMSE(resp, des)+", R2: "+Math.pow(Meuse.getPearson(resp, des),2.0));					
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
										
					sdCoeff += Math.pow( beta[1] - som.matrix.get(p)[0][0], 2 );
					sdInter += Math.pow( beta[0] - som.output.get(p)[0], 2 );
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
		
		Sorter<double[]> sorter = new DefaultSorter<double[]>(fDist);
		LLMNG ng = new LLMNG(grid.getPositions().size(), grid.getSizeOfDim(0), 0.01, 0.5, 0.005, 
				grid.getSizeOfDim(0), 0.01, 0.5, 0.005, 
				sorter, new int[]{0}, samples.get(0).length, 1);
						
		for (int t = 0; t < 100000; t++) {
			int j = r.nextInt(samples.size());
			ng.train( (double)t/T_MAX, samples.get(j), desired.get(j) );
		}
		
		List<double[]> res = new ArrayList<double[]>();
		for (double[] x : samples)
			res.add(ng.present(x));
		
		sdCoeff = 0;
		sdInter = 0;
		c = 0;
		Map<double[],Set<double[]>> m2 = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
		// ------- print stats ------- 
		for (double[] p : ng.getNeurons() ) {
			Set<double[]> s = m2.get(p);
			if (s.isEmpty())
				continue;
			
			//log.debug("----------------------");
			//log.debug(p + "," + p.hashCode()+","+ s.size());
			//log.debug("p: " + Arrays.toString(p));
			{
				{ // LLM
					List<double[]> des = new ArrayList<double[]>();
					List<double[]> resp = new ArrayList<double[]>();
					for( double[] d : s ) {
						resp.add( som.present(d));
						des.add( desired.get(samples.indexOf(d)));
					}
					//log.debug("LLM o: " + Arrays.toString( ng.output.get(p))+ ", m:" + Arrays.toString(ng.matrix.get(p)[0]) );
					//log.debug("RMSE: "+Meuse.getRMSE(resp, des)+", R2: "+Math.pow(Meuse.getPearson(resp, des),2.0));					
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
							x[l][i] -= p[i];
						l++;
					}
				
					OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
					ols.setNoIntercept(false);
					ols.newSampleData(y, x);
					double[] beta = ols.estimateRegressionParameters();
										
					sdCoeff += Math.pow( beta[1] - ng.matrix.get(p)[0][0], 2 );
					sdInter += Math.pow( beta[0] - ng.output.get(p)[0], 2 );
					//llm.output.put(p,new double[]{ beta[0]} );
					//llm.matrix.put(p, new double[][]{ {beta[1] } });
					
					//((log.debug("OLS B:"+Arrays.toString(beta) );
					
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
					//log.debug("RMSE: "+Meuse.getRMSE(resp, des)+", R2: "+Math.pow(Meuse.getPearson(resp, des),2.0));
				}
			}
			c++;
		}

		log.debug("------ NG: --------");
		log.debug("SDs: "+Math.sqrt(sdInter/c)+", "+Math.sqrt(sdCoeff/c));
		log.debug("RMSE: "+Meuse.getRMSE(res, desired));
		log.debug("R^2: "+Math.pow(Meuse.getPearson(res, desired), 2));
		
	}
}

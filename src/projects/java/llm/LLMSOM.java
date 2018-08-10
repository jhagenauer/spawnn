package llm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.SupervisedNet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class LLMSOM extends SOM implements SupervisedNet {

	private static Logger log = Logger.getLogger(LLMSOM.class);

	public Map<GridPos, double[]> output = new HashMap<GridPos, double[]>(); // intercept
	public Map<GridPos, double[][]> matrix = new HashMap<GridPos, double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private KernelFunction nb2;
	private DecayFunction lr2;
	private int[] fa; //indices of independent variables
		
	public LLMSOM( KernelFunction nb, DecayFunction lr, 
			Grid<double[]> grid, BmuGetter<double[]> bmuGetter, 
			KernelFunction nb2, DecayFunction lr2,
			int[] fa, int outDim) {
		super(nb, lr, grid, bmuGetter);
		Random r = new Random(1);
		
		if( fa == null ) {
			this.fa = new int[grid.getPrototypes().iterator().next().length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = fa;
		
		for (GridPos d : grid.getPositions()) {
			// init output
			double[] o = new double[outDim];
			for (int i = 0; i < o.length; i++)
				o[i] = r.nextDouble();
			output.put(d, o);

			// init matrices
			double[][] m = new double[outDim][this.fa.length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
		this.nb2 = nb2;
		this.lr2 = lr2;
	}

	public double[] present(double[] x) {
		GridPos bmuPos = bmuGetter.getBmuPos(x, grid);
		return getResponse(x, bmuPos);
	}

	public double[] getResponse( double[] x, GridPos gp ) {
		double[] w = grid.getPrototypeAt(gp);
		double[][] m = matrix.get(gp);
		double[] r = new double[m.length]; // calculate product m*diff
		for (int i = 0; i < m.length; i++) // row, outputs
			for (int j = 0; j < m[i].length; j++) // column, inputs
				r[i] += m[i][j] * (x[fa[j]]-w[fa[j]]);

		// add output
		for (int i = 0; i < r.length; i++)
			r[i] += output.get(gp)[i];

		return r;
	}
	
	@Override
	public double[] getResponse(double[] x, double[] neuron) {
		return getResponse(x, grid.getPositionOf(neuron) );
	}
	
	public boolean aMode = true;
	public boolean uMode = false;

	public void train(double t, double[] x, double[] desired) {
		
		GridPos bmuPos = bmuGetter.getBmuPos(x, grid);		
		for (GridPos p : grid.getPositions()) { 
			double[] w = grid.getPrototypeAt(p);	
			double[] r = getResponse(x, p); 
			
			double adapt = nb.getValue(grid.dist(bmuPos, p), t) * lr.getValue(t);

			double[] v = grid.getPrototypeAt(p);
			for (int j = 0; j < v.length; j++)
				v[j] = v[j] + adapt * (x[j] - v[j]);
			grid.setPrototypeAt(p, v);
			
			if( uMode )
				r = getResponse(x, p); 
			
			// adapt output
			double adapt2 = nb2.getValue(grid.dist(bmuPos, p), t) * lr2.getValue(t);
			double[] o = output.get(p); // output Vector
			for (int i = 0; i < desired.length; i++) {
				if( aMode ) // Fritzke
					o[i] += adapt2 * (desired[i] - o[i]);
				else if( aMode ) // Martinetz
					o[i] += adapt2 * (desired[i] - r[i]);
			}
						
			// adapt matrix
			double[][] m = matrix.get(p);
			for (int i = 0; i < m.length; i++) // row
				for (int j = 0; j < m[i].length; j++) // column
					m[i][j] += adapt2 * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
		}
	}
	
	public void setBmuGetter( BmuGetter<double[]> bmuGetter ) {
		this.bmuGetter = bmuGetter;
	}
	
	public BmuGetter<double[]> getBmuGetter() {
		return this.bmuGetter;
	}
	
	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		List<double[]> all = DataUtils.readCSV("data/polynomial.csv");
		DataUtils.zScore(all);
		final Dist<double[]> dist = new EuclideanDist();

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : all) {
			samples.add(new double[] { d[0], d[1], d[2], d[3], d[4] });
			desired.add(new double[] { d[5] });
		}
		log.debug(samples.size());
		
		log.debug("------------ LM: ----------");
		
		try { // predict for rmse and r2
			double[] y = new double[desired.size()];
			for (int i = 0; i < desired.size(); i++)
				y[i] = desired.get(i)[0];

			double[][] x = new double[samples.size()][];
			for (int i = 0; i < samples.size(); i++)
				x[i] = samples.get(i);
			
			// training
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(false);
			ols.newSampleData(y, x);
			double[] beta = ols.estimateRegressionParameters();
			
			// testing
			List<double[]> response = new ArrayList<double[]>();
			List<double[]> desiredResponse = new ArrayList<double[]>();
			for (int i = 0; i < samples.size(); i++) {
				double[] xi = samples.get(i);

				double p = beta[0]; // intercept at beta[0]
				for (int j = 1; j < beta.length; j++)
					p += beta[j] * xi[j - 1];

				response.add(new double[] { p });
				desiredResponse.add(desired.get(i));
			}
			log.debug("RMSE: "+Meuse.getRMSE(response, desiredResponse));
			log.debug("R^2: "+Math.pow(Meuse.getPearson(response, desiredResponse), 2));
		} catch (SingularMatrixException e) {
			log.debug(e.getMessage());
		}
		
		log.debug("----------- LLMSOM: --------------");

		Grid2D<double[]> grid = new Grid2DHex<double[]>(8, 6);
		SomUtils.initRandom(grid, samples);
		BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(dist);
		
		KernelFunction nb1 = new GaussKernel(new LinearDecay(10, 1));
		DecayFunction lr1 = new LinearDecay(1, 0.0);
		KernelFunction nb2 = new GaussKernel(new LinearDecay(10, 1));
		DecayFunction lr2 = new LinearDecay(1, 0.0);
		
		LLMSOM llm = new LLMSOM(nb1,lr1, grid, bmuGetter, nb2, lr2, new int[]{0,1,2,3,4}, 1);

		for (int t = 0; t < 100000; t++) {
			int j = r.nextInt(samples.size());
			llm.train((double) t / T_MAX, samples.get(j), desired.get(j));
		}

		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples)
			response.add(llm.present(x));

		log.debug("RMSE: "+Meuse.getRMSE(response, desired));
		log.debug("R^2: "+Math.pow(Meuse.getPearson(response, desired), 2));
		printTopoCorrelations(samples, grid, bmuGetter, llm.matrix, dist );
		
		log.debug("----------- SOM + LM: -----------");
		Map<GridPos,Set<double[]>> bmuMapping = SomUtils.getBmuMapping(samples, grid, bmuGetter);
		Map<GridPos,double[][]> m = new HashMap<GridPos,double[][]>();
		
		response = new ArrayList<double[]>();
		List<double[]> desiredResponse = new ArrayList<double[]>();
		for( GridPos p : bmuMapping.keySet() ) {
			Set<double[]> s = bmuMapping.get(p);
			double[] y = new double[s.size()];
			double[][] x = new double[s.size()][];
			int l = 0;
			for( double[] d : s ) {
				int idx = samples.indexOf(d);
				y[l] = desired.get(idx)[0];
				x[l] = samples.get(idx);
				l++;
			}
				
			// training
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(false);
			ols.newSampleData(y, x);
			double[] beta = ols.estimateRegressionParameters();
			
			// testing
			for (int i = 0; i < x.length; i++) {
				double[] xi = x[i];
				double ps = beta[0]; // intercept at beta[0]
				for (int j = 1; j < beta.length; j++)
					ps += beta[j] * xi[j - 1];

				response.add(new double[] { ps });
				desiredResponse.add( new double[]{y[i]} );
			}
			m.put( p, new double[][]{beta});
		}
		log.debug("RMSE: "+Meuse.getRMSE(response, desiredResponse));
		log.debug("R^2: "+Math.pow(Meuse.getPearson(response, desiredResponse), 2));
		printTopoCorrelations(samples, grid, bmuGetter, m, dist );
		
	}
	
	private static void printTopoCorrelations( List<double[]> samples, Grid<double[]> grid, BmuGetter<double[]> bmuGetter, Map<GridPos, double[][]> m, Dist<double[]> dist ) {			
		Map<double[],GridPos> map = new HashMap<double[],GridPos>();
		for( double[] d : samples )
			map.put(d, bmuGetter.getBmuPos(d, grid) );
				
		int size = (samples.size()*(samples.size()-1))/2;
		int outDim = m.values().iterator().next().length;
		double[] gridDist = new double[size];
		double[] vectorDist = new double[size];
		double[][] mDist = new double[outDim][size];
		int index = 0;
		for( int i = 0; i < samples.size()-1; i++) {
			for( int j = i+1; j < samples.size(); j++ ) {
				double[] v1 = samples.get(i);
				double[] v2 = samples.get(j);
				GridPos p1 = map.get(v1);
				GridPos p2 = map.get(v2);
													
				vectorDist[index] = dist.dist(v1, v2);
				gridDist[index] = grid.dist( p1, p2 );
				for( int l = 0; l < outDim; l++ )
					mDist[l][index] = dist.dist(m.get(p1)[l], m.get(p2)[l]);
				index++;
			}
		}
					
		if( index != size )
			throw new RuntimeException("Error in array length caclulation in pearson correlation!");
		
		{ // Pearson
			log.debug("Pearson:");
			PearsonsCorrelation cor = new PearsonsCorrelation();
			log.debug("proto-topo: "+cor.correlation(gridDist,vectorDist));
			for( int l = 0; l < outDim; l++ )
				log.debug("m-topo "+l+": "+cor.correlation(gridDist, mDist[l]));
		} 
		{ // spearman
			log.debug("Spearman:");
			SpearmansCorrelation cor = new SpearmansCorrelation();
			log.debug("proto-topo:"+cor.correlation(gridDist,vectorDist));
			for( int l = 0; l < outDim; l++ )
				log.debug("m-topo "+l+": "+cor.correlation(gridDist, mDist[l]));
		} 
	}
}

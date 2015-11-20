package wmng.llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.SupervisedNet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.SorterContext;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class ContextNG_LLM extends NG implements SupervisedNet {
	
	private static Logger log = Logger.getLogger(ContextNG_LLM.class);
	
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private boolean useContext;
	private int[] fa;
	private DecayFunction neighborhood2, adaptation2;
	
	public ContextNG_LLM( List<double[]> neurons, 
			DecayFunction neighborhood, DecayFunction adaptation, 
			DecayFunction neighborhood2, DecayFunction adaptation2, 
			SorterContext sorter, int[] fa, int outDim, boolean useContext ) {
		super(neurons,neighborhood,adaptation,sorter);
		
		this.adaptation2 = adaptation2;
		this.neighborhood2 = neighborhood2;
		this.fa = fa;
		this.useContext = useContext;
		
		Random r = new Random();
		for( double[] d : getNeurons() ) {
			// init output
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			// init matrices
			double[][] m = new double[outDim][neurons.get(0).length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
	}
	
	@Override
	public double[] present(double[] x) {
		sorter.sort(x, neurons);
		return getResponse(x, getNeurons().get(0) );
	}
	
	public double alpha = 0.0;

	@Override
	public double[] getResponse(double[] x, double[] w) {
		double[][] m = matrix.get(w);
		double[] r = new double[m.length]; // calculate product m*diff
						
		if( useContext ) {
			for( int i = 0; i < m.length; i++ ) // row, outputs
				for( int j = 0; j < fa.length; j++ ) // column, inputs
					r[i] += (1.0-alpha) * m[i][fa[j]] * (x[fa[j]] - w[fa[j]] );
			
			double[] context = ((SorterContext)sorter).getContext(x);
			for( int i = 0; i < m.length; i++ ) // row, outputs
				for( int j = 0; j < fa.length; j++ ) // column, inputs
					r[i] += (alpha) * m[i][x.length + fa[j]] * (context[fa[j]] - w[x.length + fa[j]] );
		} else {
			for( int i = 0; i < m.length; i++ ) // row, outputs
				for( int j = 0; j < fa.length; j++ ) // column, inputs
					r[i] += m[i][fa[j]] * (x[fa[j]] - w[fa[j]] );
		}
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(w)[i];
							
		return r;
	}

	@Override
	public void train(double t, double[] x, double[] desired) {
		double[] context = ((SorterContext)sorter).getContext(x); // sort affects context
		sortNeurons(x);
					
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
		double l2 = neighborhood2.getValue(t);
		double e2 = adaptation2.getValue(t);
			
		// adapt
		for (int k = 0; k < neurons.size(); k++) {
			double[] w = neurons.get(k);
			double[] r = getResponse( x, w ); 
			
			double adapt = e * Math.exp((double) -k / l);			
			// adapt weights
			for (int i = 0; i < x.length; i++)
				w[i] += adapt * (x[i] - w[i]);
			
			// adapt context vector part
			if( context != null )
				for( int i = 0; i < context.length; i++ )
					w[x.length + i] += adapt * (context[i] - w[x.length + i]);
			
			// -------------------------------------------------------------------------
			
			// adapt output
			double adapt2 = e2 * Math.exp( -(double)k/l2 );
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ )
				o[i] += adapt2 * (desired[i] - o[i]);
							
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < x.length; j++ ) // column, attributes
					m[i][j] += adapt2 * (desired[i] - r[i]) * (x[j] - w[j]); // outer product
			
			// adapt context-part of matrix
			if( useContext ) {
				for( int i = 0; i < m.length; i++ )  // row
					for( int j = 0; j < x.length; j++ ) // column, attributes
						m[i][x.length + j] += adapt2 * (desired[i] - r[i]) * (context[j] - w[x.length + j]); // outer product
			}
		}		
	}
	
	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), new int[] { 0,1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length-1);
					
			samples.add(nd);
			desired.add(new double[]{d[d.length-1]});
		}
						
		final int[] fa = new int[samples.get(0).length-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
		
		final Map<double[], Map<double[], Double>> dMap = getDistanceMatrix(samples, gDist);
				
		for( final int T_MAX : new int[]{ 40000 } )	
		for( final int nrNeurons : new int[]{ 8 } )
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 0.1 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.01 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.01 })
		for( final boolean useContext : new boolean[]{ false, true } )
		for( final double alpha : new double[]{0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0 } )
		for( final double beta : new double[]{ 0.0 } )
		{
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 64; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> samplesTrain = new ArrayList<double[]>(samples);
						List<double[]> desiredTrain = new ArrayList<double[]>(desired);
						List<double[]> samplesVal = new ArrayList<double[]>();
						List<double[]> desiredVal = new ArrayList<double[]>();
						
						while( samplesVal.size() < samples.size()/3 ) {
							int idx = r.nextInt(samplesTrain.size());
							samplesVal.add(samplesTrain.remove(idx));
							desiredVal.add(desiredTrain.remove(idx));
						}
						Map<double[], Map<double[], Double>> dMapTrain = getDistanceMatrix(samplesTrain, gDist);
						
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] rs = samplesTrain.get(r.nextInt(samplesTrain.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samplesTrain)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);
						
						DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						
						ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1, useContext);
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samplesTrain.size());
							ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
						}
						ng.alpha = alpha;
						
						sorter.setWeightMatrix(dMap); // full weight-matrix
						for( int i = 0; i < 1000; i++ ) // update/prepare histMap
							for( double[] x : samplesVal )
								sorter.sort(x, neurons);
												
						List<double[]> responseVal = new ArrayList<double[]>();
						for (double[] x : samplesVal)
							responseVal.add(ng.present(x));				
																									
						return new double[] { 
								Meuse.getRMSE(responseVal, desiredVal)
								};
					}
				}));
			}
			es.shutdown();
			
			DescriptiveStatistics ds[] = null;
			for (Future<double[]> ff : futures) {
				try {
					double[] ee = ff.get();
					if (ds == null) {
						ds = new DescriptiveStatistics[ee.length];
						for (int i = 0; i < ee.length; i++)
							ds[i] = new DescriptiveStatistics();
					}
					for (int i = 0; i < ee.length; i++)
						ds[i].addValue(ee[i]);
				} catch (InterruptedException ex) {
					ex.printStackTrace();
				} catch (ExecutionException ex) {
					ex.printStackTrace();
				}
			}
						
			try {
				String fn = "output/resultContextNG_LLM.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,alpha,beta,useContext,rmse\n").getBytes());
				}
				String s = T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+useContext+","+alpha+","+beta;
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static Map<double[],Map<double[],Double>> getDistanceMatrix( List<double[]> samples, Dist<double[]> gDist ) {
		return GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getKNNs(samples,gDist,8,false)));
	}
}

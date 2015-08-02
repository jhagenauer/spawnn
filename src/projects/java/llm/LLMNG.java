package llm;

import java.util.ArrayList;
import java.util.Collections;
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
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

public class LLMNG extends NG {
	
	private static Logger log = Logger.getLogger(LLMNG.class);
		
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private double lInit2, lFinal2, eInit2, eFinal2;
	private int[] fa;
	
	@Deprecated
	public LLMNG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, Sorter<double[]> bg, int outDim ) {
		this(numNeurons,lInit,lFinal,eInit,eFinal,lInit,lFinal,eInit,eFinal, bg, null, dim, 1);
	}
	
	public LLMNG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal,
			double lInit2, double lFinal2, double eInit2, double eFinal2, 
			Sorter<double[]> bg, int[] fa, int inDim, int outDim ) {
		super(numNeurons,lInit,lFinal,eInit,eFinal,inDim,bg);
		
		this.lInit2 = lInit2;
		this.lFinal2 = lFinal2;
		this.eInit2 = eInit2;
		this.eFinal2 = eFinal2;
		
		if( fa == null ) {
			this.fa = new int[inDim];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = fa;

		Random r = new Random();
		for( double[] d : getNeurons() ) {
			// init output
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			// init matrices
			double[][] m = new double[outDim][this.fa.length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
	}
	
	public double[] present( double[] x ) {
		sortNeurons(x);
		return getResponse(x, getNeurons().get(0) );
	}
	
	public double[] getResponse( double[] x, double[] neuron ) {			
		double[][] m = matrix.get(neuron);
		double[] r = new double[m.length]; // calculate product m*diff
		
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < m[i].length; j++ ) // column, inputs
				r[i] += m[i][j] * (x[fa[j]] - neuron[fa[j]]);
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(neuron)[i];
							
		return r;
	}
	
	public void train( double t, double[] x, double[] desired ) {
		train(t, x); // assumes that neurons are sorted after this call
		
		double l = lInit2 * Math.pow( lFinal2/lInit2, t );
		double e = eInit2 * Math.pow( eFinal2/eInit2, t );
				
		for( int k = 0; k < getNeurons().size(); k++ ) {
			double adapt = e * Math.exp( -(double)k/l );
			double[] w = getNeurons().get(k);
			
			// double[] r = getResponse( x, w ); 
			
			// adapt output
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ ) { 
				o[i] += adapt * (desired[i] - o[i]); // martinetz
				//o[i] += adapt * (desired[i] - r[i]); // fritzke
			}
			
			double[] r = getResponse( x, w );
			
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < m[i].length; j++ ) // column
					m[i][j] += adapt * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
		}
	}
	
	public void setSorter( Sorter<double[]> sorter ) {
		this.sorter = sorter;
	}
	
	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		List<double[]> all = DataUtils.readCSV("data/polynomial.csv" );
		DataUtils.zScore(all);
		final Dist<double[]> dist = new EuclideanDist();
		
		final int maxK = 10;
		
		ExecutorService es = Executors.newFixedThreadPool( 4 );
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		for (int j = 0; j < 32; j++) {
			Collections.shuffle(all);
			final List<double[]> samples = new ArrayList<double[]>();
			final List<double[]> desired = new ArrayList<double[]>();

			for( double[] d : all ) {	
				samples.add( new double[]{ d[0], d[1], d[2], d[3], d[4] });
				desired.add(new double[] { d[5] });
			}
			
			for (int k = 0; k < maxK; k++) {
				final int K = k;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						List<double[]> training = new ArrayList<double[]>();
						List<double[]> trainingDesired = new ArrayList<double[]>();
						List<double[]> validation = new ArrayList<double[]>();
						List<double[]> validationDesired = new ArrayList<double[]>();

						for (int i = 0; i < samples.size(); i++) {
							if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
								validation.add(samples.get(i));
								validationDesired.add(desired.get(i));
							} else {
								training.add(samples.get(i));
								trainingDesired.add(desired.get(i));
							}
						}
		
						Sorter<double[]> s = new DefaultSorter<double[]>(dist);
						LLMNG llm = new LLMNG(12, 6, 0.01, 0.5, 0.005, 
								6, 0.01, 0.5, 0.005, 
								s, new int[]{0,1,2,3,4}, samples.get(0).length, 1);
										
						for (int t = 0; t < 100000; t++) {
							int j = r.nextInt(training.size());
							llm.train( (double)t/T_MAX, training.get(j), desired.get(j) );
						}
						
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : validation)
							response.add(llm.present(x));
											
						return new double[]{
								Meuse.getRMSE(response, validationDesired),
								Math.pow(Meuse.getPearson(response, validationDesired), 2)
						};
					}
				}));
			}
		}
	
		es.shutdown();

		DescriptiveStatistics[] ds = null;

		for (Future<double[]> f : futures) {
			try {
				double[] d = f.get();

				if (ds == null) {
					ds = new DescriptiveStatistics[d.length];
					for (int i = 0; i < d.length; i++)
						ds[i] = new DescriptiveStatistics();
				}

				for (int i = 0; i < d.length; i++)
					ds[i].addValue(d[i]);
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		String desc = "";
					
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < ds.length; i++)
			sb.append(ds[i].getMean() + ",");
		log.debug(desc+","+sb.substring(0, Math.min(sb.length(),500)));
	}
}

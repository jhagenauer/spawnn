package context.time.discrete;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import context.time.TimeSeries;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetterTimeMSOM;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.net.SOM;
import spawnn.som.net.ContextSOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import cern.colt.bitvector.BitVector;

public class TimeSeriesDiscrete {

	private static Logger log = Logger.getLogger(TimeSeriesDiscrete.class);

	public static void main(String[] args) {
				
		Dist<double[]> fDist = new EuclideanDist( new int[]{1});
		
		List<double[]> samples = null;
		try {		
			samples = DataUtils.readCSV(new FileInputStream("data/somsd/binary.csv") ).subList(0, 1000000);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		int T_MAX = samples.size();		
		int rcpFieldSize = 30;
		
		XYSeriesCollection dataset = new XYSeriesCollection();
						
		{ // SOM
			/*log.debug("som");
			span.som.bmu.BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(fDist);
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10,10);
			SomUtils.initRandom(grid, samples);
								
			Som som = new Som( new Gaussian(grid.getMaxDist()), new LinearRate(1.0,0.0), grid, bg );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get( t % samples.size() );
				som.train((double) t / T_MAX, x);
			}
					
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
			double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
			
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("som");		
			for( int size = 0; size < tqe.length; size ++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			
			log.debug("--------------");*/
		}
				
		{ // SOMSD, to beat: 4.647
			
			/*log.debug("somsd entropy");
			
			// best mg: 500, 0.005
			for( int maxAddedSize : new int[]{ 500 } ) { 
				for( double eta : new double[]{ 0.05 } ) {
					
					
					Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
					
					// init grid
					Random r = new Random();
					for( GridPos gp : grid.getPositions() ) {
						double[] rs = samples.get( r.nextInt(samples.size() ) );
						double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
						grid.setPrototypeAt(gp, d );
					}
										
					List<double[]> added = new ArrayList<double[]>();
					double old_entropy = 0.0;
					double alpha = 1.0; // stärke von f
					
					BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, alpha ); // .96 is best ATM
													
					Som som = new SomSD( new Gaussian(grid.getMaxDist()), new LinearRate(1.0,0.0), grid, bg );
					for (int t = 0; t < T_MAX; t++) {				
										
						double[] x = samples.get( t % samples.size() );
						som.train((double) t / T_MAX, x);
						
						added.add(x);
						
						// adapt alpha
						if( added.size() == maxAddedSize ) {
							
							Map<GridPos,Set<double[]>> m = SomUtils.getBmuMapping(added, grid, bg);
							
							// shannon entropy
							double entropy = 0;
							for( GridPos gp : grid.getPositions() ) { 
								if( m.containsKey(gp) && !m.get(gp).isEmpty() ) {
									double d = (double)m.get(gp).size()/added.size();
									entropy += -d * Math.log( d )/Math.log(2);
								}
							}
														
							if( old_entropy > entropy ) {
								alpha += eta*(1.0-(double)t/T_MAX);
							} else if( old_entropy < entropy ) {
								alpha -= eta*(1.0-(double)t/T_MAX);
							}
							alpha = Math.min( Math.max(0,alpha), 1);
							
							if( entropy > 2 ) {
								log.debug(entropy+" > 2 !!!");
								//break;
							}
							
							bg.setAlpha(alpha);					
							old_entropy = entropy;
							
							added.clear();
						}
					}
					log.debug("final alpha: "+alpha);
						
					bg.setContext(null);
					Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
					double[] tqe = TimeSeries.getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
					
					double sum = 0;
					for( double d : tqe )
						sum += d;					
					log.info(maxAddedSize+","+eta+","+sum);
					
					log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
					
					XYSeries error = new XYSeries("somsd entropy");		
					for( int size = 0; size < tqe.length; size++ )
						error.add(size,tqe[size]);
					dataset.addSeries(error);
					
					log.debug("--------------");
					
				}
			}
			*/
		}
		
		{ // SOMSD dynamic
			
			/*log.debug("somsd dynamic");
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10,10);
			
			// init grid
			Random r = new Random();
			for( GridPos gp : grid.getPositions() ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
				grid.setPrototypeAt(gp, d );
			}
						
			BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, 1.0 );
										
			// best mg: lr 0.71, to=0.96
			Som som = new SomSD( new Gaussian(grid.getMaxDist()), new LinearRate(0.71,0.0), grid, bg );
			for (int t = 0; t < T_MAX; t++) {
				
				double from = 1.0;
				double to = 0.6788;
				double span = 1.0;
				
				// lin
				double f = (to - from)/span;
				double alpha = ((double)t/T_MAX) * f + from;*/
								
				// exp
				/*double eta = 0.1;
				double g;
				
				if( eta == 0.0 )
					eta = (g = Math.sqrt(to)) / (Math.sqrt(from) + g);
			    
				double c = (eta * eta * (from - to) - to + 2. * eta * to) / (2. * eta - 1.);
			    double f = Math.log((from - c) / (to - c)) / span;
			    g = Math.log(from - c) / f;

				double alpha =  c + Math.exp(f * (g - (double)t/T_MAX));*/
								
				/*bg.setAlpha( alpha );
								
				double[] x = samples.get( t % samples.size() );
				som.train((double) t / T_MAX, x);
			}
			
			bg.setContext(null);
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
			double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
			
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("somsd dynamic");		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			
			log.debug("--------------");*/
		}
					
		{ // MSOM
			for( double alpha : new double[]{ 0.03 } ) { // 1 -> 2 stück weit auseinander, 0 -> alles auf einem
								
				log.debug("msom "+alpha);
				Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
				
				// init grid
				Random r = new Random();
				for( GridPos gp : grid.getPositions() ) {
					double[] rs = samples.get( r.nextInt(samples.size() ) );
					double[] d = Arrays.copyOf(rs, rs.length * 2 );
					
					
					grid.setPrototypeAt(gp, d );
				}
											
				BmuGetterTimeMSOM bg = new BmuGetterTimeMSOM(fDist, alpha, 0.45 );
				KernelFunction nbKernel = new GaussKernel( new LinearDecay( 10, 1 ) );
				DecayFunction learningRate = new LinearDecay( 1.0, 0.0 );
				
				int maxAdded = 800;
				double eta = 0.2;
				List<double[]> added = new ArrayList<double[]>();
				double old_entropy = 0.0;
								
				SOM som = new ContextSOM( nbKernel, learningRate , grid, bg, samples.get(0).length );
				for (int t = 0; t < T_MAX; t++) {
					
					//bg.setAlpha( 0 );
										
					double[] x = samples.get( t % samples.size() );
					som.train((double) t / T_MAX, x);
					
					added.add(x);
					
					// adapt alpha
					if( added.size() == maxAdded ) {
						
						Map<GridPos,Set<double[]>> m = SomUtils.getBmuMapping(added, grid, bg);
						double entropy = SomUtils.getEntropy(added, m);		
													
						if( old_entropy > entropy ) {
							alpha -= eta*(1.0-(double)t/T_MAX);
						} else if( old_entropy < entropy ) {
							alpha += eta*(1.0-(double)t/T_MAX);
						}
						alpha = Math.min( Math.max(0,alpha), 1);
						
						//if( entropy > 2 ) {
							log.debug(entropy+"\t"+alpha+"\t"+t);
							//break;
						//}
						
						bg.setAlpha(alpha);					
						old_entropy = entropy;
						
						added.clear();
					}					
				}
				log.debug("final alpha: "+alpha);
								
				bg.setLastBmu(null);
				Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);				
				double[] tqe = TimeSeries.getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );

				log.debug("size: "+bmus.size() );
				for( GridPos gp : bmus.keySet() )
					if( !bmus.get(gp).isEmpty() )
						System.out.println(gp+";"+bmus.get(gp).size());
								
				double sum = 0;
				for( double d : tqe )
					sum += d;
				log.debug("sum: "+sum);
				
				log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
								
				XYSeries error = new XYSeries("somsd "+alpha);		
				for( int size = 0; size < tqe.length; size++ )
					error.add(size,tqe[size]);
				dataset.addSeries(error);

				log.debug("-----------------");
				
				Map<GridPos,List<List<double[]>>> bmuSeqs = TimeSeries.getReceptiveFields(samples, bmus, rcpFieldSize);
				Map<GridPos, List<Double>> rf = getIntersectReceptiveFields(bmuSeqs, 1);
															
				// convert seqs to bitvectors
				Map<GridPos,BitVector> m = new HashMap<GridPos,BitVector>();
				for( GridPos p : rf.keySet() ) {
					List<Double> l = new ArrayList<Double>(rf.get(p));
					Collections.reverse(l);
														
					BitVector bv = new BitVector(l.size());
					for( int i = 0; i < l.size(); i++ )
						bv.put(i, l.get(i) > 0.0);
					m.put(p, bv);
					
					log.debug(BinaryAutomaton.toString(bv));
				}
				
				Set<BitVector> set = new HashSet<BitVector>(m.values());
				List<String> nodes = new ArrayList<String>( BinaryAutomaton.getNodes( set, 0, new double[]{ 0, 0 }, new BitVector(0), 10 ) );
				
				BufferedWriter bw = null;
				try {
					bw = new BufferedWriter( new FileWriter("output/tree.dat"));
					for( String s : nodes )
						bw.write(s);
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					try { bw.close(); } catch( Exception e ) {}
				}
			}
		}
				               
		JFreeChart lineplot = ChartFactory.createXYLineChart("","past index","Error",dataset,PlotOrientation.VERTICAL,true,true,false);
		try {
			ChartUtilities.saveChartAsPNG(new File("output/lineplot_time.png"), lineplot, 1024, 768);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	public static <T> Map<T, List<Double>> getIntersectReceptiveFields(Map<T, List<List<double[]>>> bmuSeqs, int idx) {
		Map<T, List<Double>> m = new HashMap<T, List<Double>>();

		for (T bmu : bmuSeqs.keySet()) {

			int minLength = Integer.MAX_VALUE;
			for (List<double[]> l : bmuSeqs.get(bmu))
				minLength = Math.min(l.size(), minLength);

			List<Double> rField = new ArrayList<Double>();

			for (int i = 0; i < minLength; i++) {

				double prev = 0;
				boolean ident = true;
				for (int j = 0; j < bmuSeqs.get(bmu).size() && ident; j++) {
					List<double[]> l = bmuSeqs.get(bmu).get(j);
					double c = l.get(l.size() - 1 - i)[idx];
					if (j > 0 && c != prev)
						ident = false;
					prev = c;
				}

				if (ident)
					rField.add(prev);
				else
					break;
			}

			Collections.reverse(rField);
			m.put(bmu, rField);
		}
		return m;
	}
	
	public static BitVector doubleListToBV( List<Double> l ) {
		BitVector bv = new BitVector(l.size());
		for( int i = 0; i < l.size(); i++ )
			bv.put(i, l.get(i) > 0.0);
		return bv;
	}
	
	public static void saveGnuplotTree( Set<BitVector> rf, String fn ) {	
		List<String> nodes = new ArrayList<String>( BinaryAutomaton.getNodes( rf, 0, new double[]{ 0, 0 }, new BitVector(0), 10 ) );
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter( fn ));
			for( String s : nodes )
				bw.write(s);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try { bw.close(); } catch( Exception e ) {}
		}
	}
	
	public static String toString(BitVector bv ) {
		StringBuffer sb = new StringBuffer();
		for( int i = 0; i < bv.size(); i++ )
			if( bv.get(i) )
				sb.append("1");
			else
				sb.append("0");
		return sb.toString();
	}
}

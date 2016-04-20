package context.time;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterMNG;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TimeSeries {

	private static Logger log = Logger.getLogger(TimeSeries.class);
	
	public static void main(String[] args) {
				

		Random r = new Random();
		final int[] fa = new int[]{1};
		Dist<double[]> fDist = new EuclideanDist( fa );
		//Dist tDist = new SubDist(eDist, new int[]{0});
		
		List<double[]> samples = null;
		try {		
			samples = DataUtils.readCSV(new FileInputStream("data/mg/mgsamples.csv") ).subList(0, 150000);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		int T_MAX = samples.size();
				
		int rcpFieldSize = 60;
		
		XYSeriesCollection dataset = new XYSeriesCollection();
						
		/*{ // SOM
			log.debug("som");
			spawnn.som.bmu.BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(fDist);
			Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
			SomUtils.initRandom(grid, samples);
								
			SOM som = new SOM( new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0,0.0), grid, bg );
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
			
			log.debug("--------------");
		}*/
		
		{ // NG
			log.debug("ng");
			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < 100; i++) {
				double[] d = samples.get(r.nextInt(samples.size()));
				neurons.add(Arrays.copyOf(d, d.length));
			}
									
			Sorter<double[]> sorter = new DefaultSorter<>( fDist );
			DecayFunction nbRate = new PowerDecay(50.0, 0.01);
			DecayFunction lrRate = new PowerDecay(0.5, 0.005);
			
			NG ng = new NG(neurons, nbRate, lrRate, sorter );

			for (int t = 0; t < T_MAX; t++) {
				int j = r.nextInt(samples.size());
				ng.train((double) t / T_MAX, samples.get(j) );
			}
					
			Map<double[],Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), sorter);
			double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
			
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("ng");		
			for( int size = 0; size < tqe.length; size ++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			
			log.debug("--------------");
		}
		
		{ // SIMPLE TIME SOM
			
			/*for( double alpha = 0.974; alpha <= 0.974; alpha += 0.001 ) {// 0.974
				log.debug("simple time som "+alpha);
				span.som.bmu.BmuGetter<double[]> bg = new BmuGetterTimeSimple(fDist, alpha );
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
				
				XYSeries error = new XYSeries("simple time som "+alpha);		
				for( int size = 0; size < tqe.length; size ++ )
					error.add(size,tqe[size]);
				dataset.addSeries(error);
				
				log.debug("--------------");
			}*/
		}
		
		{ // SOMSD, to beat: 4.647
			
			/*log.debug("somsd entropy");
			
			// best mg: 500, 0.005
			for( int maxAddedSize : new int[]{ 500 } ) { 
				for( double eta : new double[]{ 0.005 } ) {
					
					
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
													
					Som som = new SomSD( new GaussKernel(grid.getMaxDist()), new LinearDecay(1.0,0.0), grid, bg );
					for (int t = 0; t < T_MAX; t++) {				
										
						double[] x = samples.get( t % samples.size() );
						som.train((double) t / T_MAX, x);
						
						added.add(x);
						
						// adapt alpha
						if( added.size() == maxAddedSize ) {
							
							Map<GridPos,Set<double[]>> m = SomUtils.getBmuMapping(added, grid, bg);
							double entropy = SomUtils.getEntropy(added, m);		
														
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
					double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
					
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
			}*/
			
		}
				
		{ // SOMSD static
			/*log.debug("somsd static");
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10,10);
			
			// init grid
			Random r = new Random();
			for( GridPos gp : grid.getPositions() ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
				grid.setPrototypeAt(gp, d );
			}
						
			BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, 0.96, samples.get(0).length, grid.getNumDimensions() );
											
			Som som = new SomSD( new Gaussian(grid.getMaxDist()), new LinearRate(0.54,0.0), grid, bg );
			for (int t = 0; t < T_MAX; t++) {
				
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
			
			XYSeries error = new XYSeries("somsd static");		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);*/
		}
		
		/*{ // SOMSD 
			
			log.debug("somsd");
			double alpha = 0.06;
			Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
			
			// init grid
			for( GridPos gp : grid.getPositions() ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
				grid.setPrototypeAt(gp, d );
			}
					
			BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, alpha );
											
			SOM som = new ContextSOM( new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0,0.0), grid, bg, samples.get(0).length );
			for (int t = 0; t < T_MAX; t++) {
				
				double[] x = samples.get( t % samples.size() );
				som.train((double) t / T_MAX, x);
			}
			
			bg.setLastBmu(null);
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
			double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
			
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("somsd "+alpha);		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			
			log.debug("--------------");
		}*/
		
		{ // SOMSD 
			/*for( double alpha = 1; alpha >= 0.90; alpha -= 0.01 ) {
				log.debug("somsd "+alpha);
				Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
				
				// init grid
				Random r = new Random();
				for( GridPos gp : grid.getPositions() ) {
					double[] rs = samples.get( r.nextInt(samples.size() ) );
					double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
					grid.setPrototypeAt(gp, d );
				}
							
				BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, alpha );
												
				KernelFunction nb = new GaussKernel( new LinearDecay(10, 1) );
				DecayFunction lr = new LinearDecay( 1.0, 0.0 );
				
				Som som = new SomSD( nb, lr , grid, bg );
				for (int t = 0; t < T_MAX; t++) {
					
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
				
				XYSeries error = new XYSeries("somsd "+alpha);		
				for( int size = 0; size < tqe.length; size++ )
					error.add(size,tqe[size]);
				dataset.addSeries(error);
				
				log.debug("--------------");
			}*/
		}
								
		{ // weighted SOM 2d
			/*for( double i = 0; i < 1 ; i+=0.1 ) {
				log.debug("weighted som 2d "+i);
				
				Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
				m.put(fDist, i);
				m.put(tDist, 1 - i);
				Dist<double[]> wd = new WeightedDist<double[]>(m);
				
				span.som.bmu.BmuGetter<double[]> bg = new span.som.bmu.DefaultBmuGetter<double[]>( wd );
				Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10,10);
				SomUtils.initRandom(grid, samples);
						
				Som som = new Som( new Gaussian(grid.getMaxDist()), new LinearRate(0.5,0.0), grid, bg );
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get( t % samples.size() );
					som.train((double) t / T_MAX, x);
				}
						
				Map<GridPos,List<double[]>> meanRcpFields = getMeanRcpFields(samples, grid, bg, rcpFieldSize);
				double[] tqe = getTemporalQuantizationError(meanRcpFields, samples, grid, bg, fDist);
				
				XYSeries error = new XYSeries("wsom "+i);		
				for( int size = 0; size < tqe.length; size++ )
					error.add(size,tqe[size]);
				dataset.addSeries(error);
			}*/
		}
						
		{ // Kangas SOM 2d
			/*for( int i = 0; i <= 3; i++ ) {
				log.debug("kangas som 2d "+i);
				span.som.bmu.BmuGetter<double[]> bg = new span.som.bmu.KangasBmuGetter<double[]>( tDist, fDist, i );
				Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
				SomUtils.initRandom(grid, samples);
						
				Som som = new Som( new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0,0.0), grid, bg );
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
				
				XYSeries error = new XYSeries("Kangas "+i);		
				for( int size = 0; size < tqe.length; size ++ )
					error.add(size,tqe[size]);
				dataset.addSeries(error);
			}*/
		}
		
		// MSOM
		/*{
			log.debug("msom");
			double alpha = 0.73;
			Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
			
			// init grid
			for( GridPos gp : grid.getPositions() ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				double[] d = Arrays.copyOf(rs, rs.length*2 );
				grid.setPrototypeAt(gp, d );
			}
				
			BmuGetterTimeMSOM bg = new BmuGetterTimeMSOM(fDist, alpha,0.75);

			SOM som = new ContextSOM(new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bg, samples.get(0).length);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get( t % samples.size() );
				som.train((double) t / T_MAX, x);
			}
			
			bg.setLastBmu(null);
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
			double[] tqe = getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
			
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("msom "+alpha);		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			
			log.debug("--------------");
		}*/
		
		// MNG
		for( double[] p : new double[][]{ new double[]{0.94,0.75}, new double[]{0.85,0.65}, } ){
			log.debug("mng "+p[0]+" "+p[1]);
			
			List<double[]> neurons = new ArrayList<double[]>();
			for( int i = 0; i < 100; i++ ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				neurons.add( Arrays.copyOf(rs, rs.length*2 ) );
			}
			
			SorterMNG bg = new SorterMNG(fDist, p[0], p[1]);
			bg.setLastBmu(neurons.get(0));
			
			DecayFunction nbRate = new PowerDecay(50.0, 0.01);
			DecayFunction lrRate = new PowerDecay(0.5, 0.005);
			ContextNG ng = new ContextNG(neurons, nbRate, lrRate, bg );

			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get( t % samples.size() );
				ng.train((double) t / T_MAX, x);
			}
			
			Map<double[],Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
			double[] tqe = TimeSeries.getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
					
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("mng "+p[0]+" "+p[1]);		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			log.debug("--------------------");
		}
		
		// LAG
		for( int lag : new int[]{ 1, 2, 3, 4, 5, 6, 7 } ){
			log.debug("lag "+lag);
			
			List<double[]> neurons = new ArrayList<double[]>();
			for( int i = 0; i < 100; i++ ) {
				double[] rs = samples.get( r.nextInt(samples.size() ) );
				neurons.add( Arrays.copyOf(rs, rs.length+lag*rs.length ) );
			}
			
			int[] nfa = new int[fa.length+lag*fa.length];
			for( int i = 0; i < fa.length; i++ )
				for( int j = 0; j <= lag; j++ )
					nfa[i+j*fa.length] = fa[i]+j*samples.get(0).length;
			//log.debug("nfa: "+Arrays.toString(nfa));
													
			DecayFunction nbRate = new PowerDecay(50.0, 0.01);
			DecayFunction lrRate = new PowerDecay(0.5, 0.005);
			Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(nfa) );

			List<double[]> lagedSamples = getLagedSamples(samples, lag);
			NG ng = new NG(neurons, nbRate, lrRate, sorter );
			for (int t = 0; t < T_MAX; t++) {
				int j = r.nextInt(lagedSamples.size());
				ng.train((double) t / T_MAX, lagedSamples.get(j) );
			}
			
			Map<double[],Set<double[]>> lagedBmus = NGUtils.getBmuMapping(lagedSamples, ng.getNeurons(), sorter);
			
			// lagged bmus to normal
			Map<double[],Set<double[]>> bmus = new HashMap<double[],Set<double[]>>();
			for( Entry<double[],Set<double[]>> e : lagedBmus.entrySet() ) {
				Set<double[]> s = new HashSet<double[]>();
				for( double[] d : e.getValue() )
					s.add( samples.get( lagedSamples.indexOf(d) ) );
				bmus.put(e.getKey(), s);
			}
			
			double[] tqe = TimeSeries.getTemporalQuantizationError( samples, bmus, fDist, rcpFieldSize );
					
			double sum = 0;
			for( double d : tqe )
				sum += d;
			log.debug("sum: "+sum);
			
			log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
			
			XYSeries error = new XYSeries("lag "+lag);		
			for( int size = 0; size < tqe.length; size++ )
				error.add(size,tqe[size]);
			dataset.addSeries(error);
			log.debug("-------------");
			
		}
		
				               
		JFreeChart lineplot = ChartFactory.createXYLineChart("","past index","Error",dataset,PlotOrientation.VERTICAL,true,true,false);
		try {
			ChartUtilities.saveChartAsPNG(new File("output/lineplot_time.png"), lineplot, 1024, 768);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static <T> Map<T,List<List<double[]>>> getReceptiveFields( List<double[]> samples, Map<T,Set<double[]>> bmus, int rcpFieldSize ) {
		Map<T,List<List<double[]>>> bmuSeqs = new HashMap<T,List<List<double[]>>>();
		for( int i = rcpFieldSize - 1; i < samples.size(); i++ ) {
			
			double[] x = samples.get(i);
			T bmu = null;
			for( T gp : bmus.keySet() ) {
				if( bmus.get(gp).contains(x))  {
					bmu = gp;
					break;
				}
			}
												
			if( !bmuSeqs.containsKey(bmu) )
				bmuSeqs.put( bmu, new ArrayList<List<double[]>>() );
			
			List<double[]> sub = samples.subList( i - rcpFieldSize + 1 , i + 1 );
			bmuSeqs.get(bmu).add( sub );
		}
		return bmuSeqs;
	}
	
	public static <T> Map<T,List<double[]>> getMeanReceptiveField( Map<T,List<List<double[]>>> bmuSeqs ) {
		int inputDim = -1;
		int rFieldLength = -1;
		for( List<List<double[]>> l1 : bmuSeqs.values() ) {  
			for( List<double[]> l2 : l1 ) {
				
				if( rFieldLength > 0 && rFieldLength != l2.size() ) {
					return null;
				} else if( rFieldLength < 0 )
					rFieldLength = l2.size();
				
				if( !l2.isEmpty() ) {
					inputDim = l2.get(0).length;
					break;
				}
			if( inputDim > 0 )
				break;
			}
		}
		
				
		Map<T,List<double[]>> meanRcpFields = new HashMap<T,List<double[]>>();
		for( T bmu : bmuSeqs.keySet() ) {
			
			List<double[]> meanList = new ArrayList<double[]>();				
			for( int i = 0; i < rFieldLength; i++ ) {
		
				double[] d = new double[inputDim];
				for( List<double[]> l : bmuSeqs.get(bmu) ) 
					for( int j = 0; j < d.length; j++ )
						d[j] += l.get(i)[j]/bmuSeqs.get(bmu).size();
				
				meanList.add(d);
			}	
			meanRcpFields.put(bmu, meanList);
		}
		return meanRcpFields;
	}
	
	public static <T> double[] getTemporalQuantizationError( List<double[]> samples, Map<T,Set<double[]>> bmus, Dist<double[]> dist, int rcpFieldSize ) {
		Map<T,List<List<double[]>>> rcpFields = getReceptiveFields(samples, bmus, rcpFieldSize);
		Map<T,List<double[]>> meanRcpFields = getMeanReceptiveField(rcpFields);
		
		double[] tqe = new double[rcpFieldSize];
		for( int i = 0; i < rcpFieldSize; i++ ) {
			double sum = 0;
		
			int k = 0;
			for( int j = rcpFieldSize - 1; j < samples.size(); j++ ) {
												
				double[] x = samples.get(j);
				T bmu = null;
				for( T gp : bmus.keySet() ) {
					if( bmus.get(gp).contains(x)) {
						bmu = gp;
						break;
					}
				}
			
				List<double[]> meanSeq = meanRcpFields.get(bmu);
												
				sum += Math.pow( dist.dist( samples.get(j - i), meanSeq.get( meanSeq.size() - 1 - i ) ), 2 );
				
				k++;
			}
			//tqe[i] = sum/k;	
			tqe[i] = Math.sqrt( sum/k );	
		}
		return tqe;
	}
	
	public static List<double[]> getLagedSamples(List<double[]> samples, int lag ) {
		List<double[]> ns = new ArrayList<double[]>();
		
		for( int i = 0; i < samples.size(); i++ ) {
			double[] d = samples.get(i);
			double[] nd = new double[d.length+d.length*lag];
			for( int j = 0; j < d.length; j++ )
				nd[j] = d[j];
			
			for( int j = 1; j <= lag; j++ ) {
				if( i -j < 0 )
					continue;
				double[] l = samples.get(i - j);
				for( int k = 0; k < d.length; k++ )
					nd[k+j*d.length] = l[k];
			}			
			ns.add(nd);
		}
		return ns;
	}
}

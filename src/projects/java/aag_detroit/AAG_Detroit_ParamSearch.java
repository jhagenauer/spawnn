package aag_detroit;

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
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import spawnn.dist.AugmentedDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class AAG_Detroit_ParamSearch {
	
	enum Mode {Augmented,WNG,CNG};
	private static Logger log = Logger.getLogger(AAG_Detroit_ParamSearch.class);

	public static void main(String[] args) {
		//String inFn = "/home/julian/publications/aag_detroit/data/detroit_metro_race.shp";
		String inFn = "C://Users/hagenaj/git/aag_detroit/data/detroit_metro_race.shp";
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File(inFn), true);
		int[] fa = new int[]{0,1,2,3,4,5};
		int[] ga = new int[]{9,10};
		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);
		
		DataUtils.transform(sdf.samples, fa, Transform.zScore);
		DataUtils.zScoreGeoColumns(sdf.samples, ga, gDist);
		
		Random r = new Random(0);
		int T_MAX =	 100000;
		int nrNeurons = 96;
		
		String fn = "output/aag_detroit.csv";
		try {
			Files.write(Paths.get(fn), ("model,param,type,error\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for( Mode m : new Mode[]{ Mode.CNG, Mode.WNG } ) {
			log.debug(m);
			r.setSeed(0);
			
			List<Object> hParams = new ArrayList<Object>();
			if( m == Mode.Augmented ) {
				 for( double a = 0; a <= 4; a+=0.05 )
					 hParams.add(a);
			} else if( m  == Mode.WNG ) {
				 for( double a = 0; a <= 1.000000001; a+=0.01 )
					 hParams.add(a);
			} else {
				 for( int l = 1; l <= nrNeurons; l++ )
					 hParams.add(l);
			}
			
			for( Object o : hParams ) {
				log.debug(o);
				final Sorter<double[]> s;
				if( m == Mode.Augmented ) {
					double a = (double)o;
					Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
					s = new DefaultSorter<double[]>(aDist);
				} else if( m == Mode.WNG ) {
					double w = (double)o;
					Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
					map.put(fDist, 1 - w);
					map.put(gDist, w);
					Dist<double[]> wDist = new WeightedDist<double[]>(map);
					s = new DefaultSorter<double[]>(wDist);
				} else {
					s = new KangasSorter<double[]>(gDist, fDist, (int)o );
				}
								
				ExecutorService es = Executors.newFixedThreadPool(7);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
				
				for( int i = 0; i < 100; i++ ) {
					futures.add(es.submit(new Callable<double[]>() {
						@Override
						public double[] call() throws Exception {
							ArrayList<double[]> neurons = new ArrayList<double[]>();
							for( int j = 0; j < nrNeurons; j++ ) {
								double[] d = sdf.samples.get(r.nextInt(sdf.samples.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
										
							NG ng = new NG(neurons, nrNeurons/2, 0.01, 0.5, 0.005, s);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = sdf.samples.get(r.nextInt(sdf.samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(sdf.samples, ng.getNeurons(), s);
							return new double[]{
								DataUtils.getMeanQuantizationError(bmus, fDist),
								DataUtils.getMeanQuantizationError(bmus, gDist)
							};
						}
					}));				
				}
				es.shutdown();
				
				for( Future<double[]> f : futures ) {
					double[] d;
					try {
						d = f.get();							
						String st = m+","+o+",QE,"+d[0]+"\n"+
									m+","+o+",SQE,"+d[1]+"\n";
						Files.write(Paths.get(fn), st.getBytes(), StandardOpenOption.APPEND);
					} catch (InterruptedException | ExecutionException | IOException e) {
						e.printStackTrace();
					}
				}				
			}
		}	
	}
}

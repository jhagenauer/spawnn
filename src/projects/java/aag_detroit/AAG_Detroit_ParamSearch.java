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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
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
	
	enum Mode {Augmented,Weighted,CNG};
	private static Logger log = Logger.getLogger(AAG_Detroit_ParamSearch.class);

	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("/home/julian/publications/aag_detroit/data/detroit_metro_race.shp"), true);
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
			Files.write(Paths.get(fn), ("model,param,qe,qe_sf,sqe,sqe_sd\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for( Mode m : Mode.values() ) {
			log.debug(m);
			r.setSeed(0);
			
			List<Object> hParams = new ArrayList<Object>();
			if( m == Mode.Augmented ) {
				 for( double a = 0; a <= 4; a+=0.05 )
					 hParams.add(a);
			} else if( m  == Mode.Weighted ) {
				 for( double a = 0; a <= 1; a+=0.01 )
					 hParams.add(a);
			} else {
				 for( int l = 1; l <= nrNeurons; l++ )
					 hParams.add(l);
			}
			
			for( Object o : hParams ) {
				log.debug(o);
				Sorter<double[]> s = null;
				if( m == Mode.Augmented ) {
					double a = (double)o;
					Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
					s = new DefaultSorter<double[]>(aDist);
				} else if( m == Mode.Weighted ) {
					double w = (double)o;
					Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
					map.put(fDist, 1 - w);
					map.put(gDist, w);
					Dist<double[]> wDist = new WeightedDist<double[]>(map);
					s = new DefaultSorter<double[]>(wDist);
				} else {
					s = new KangasSorter<double[]>(gDist, fDist, (int)o );
				}
				
				DescriptiveStatistics qe = new DescriptiveStatistics();
				DescriptiveStatistics sqe = new DescriptiveStatistics();
				for( int i = 0; i < 4; i++ ) {
					
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
					qe.addValue( DataUtils.getMeanQuantizationError(bmus, fDist) );
					sqe.addValue( DataUtils.getMeanQuantizationError(bmus, gDist) );
				}
				
				String st = m+","+o+","+qe.getMean()+","+qe.getStandardDeviation()+","+sqe.getMean()+","+sqe.getStandardDeviation()+"\n";
				try {
					Files.write(Paths.get(fn), st.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
		}
		
		
	}

}

package context.time.discrete;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import context.time.TimeSeries;

import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterMNG;
import spawnn.ng.utils.NGUtils;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import cern.colt.bitvector.BitVector;

public class TimeSeriesDiscreteMNG {

	private static Logger log = Logger.getLogger(TimeSeriesDiscreteMNG.class);

	public static void main(String[] args) {
					
		/*int idx = 1;
		List<double[]> samples = null;
		try {		
			samples = DataUtils.readCSV(new FileInputStream("data/somsd/binary.csv") ).subList(0, 1000000);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}*/
			
		List<double[]> samples = DataUtils.readCSV("data/somsd/grid1x10000.csv");
		int idx = 0;
		
		EuclideanDist fDist = new EuclideanDist( new int[]{idx});
				
		int T_MAX = 150000;		
		int rcpFieldSize = 30;
		
		{ // MNG, alpha = 0.92, beta = 0.35 rocks!!!
			for( double alpha : new double[]{ 0.92 } ) {
								
				log.debug("mng "+alpha);
				SorterMNG bg = new SorterMNG(fDist, alpha, 0.35);	
				ContextNG ng = new ContextNG(100, 100/2, 0.01, 0.5, 0.005, samples.get(0).length*2, bg );
													
				for (int t = 0; t < T_MAX; t++) {				
					double[] x = samples.get( t % samples.size() );
					ng.train((double) t / T_MAX, x);						
				}
				log.debug("done.");
												
				bg.setLastBmu(null);
				
				Map<double[],Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
				
				log.debug("entr: "+SomUtils.getEntropy(samples, bmus));
				
				log.debug("-----------------");
				
				for( double[] p : bmus.keySet() )
					log.debug(Arrays.toString(p));
				
				Map<double[],List<List<double[]>>> bmuSeqs = TimeSeries.getReceptiveFields(samples, bmus, rcpFieldSize);
				Map<double[], List<Double>> rf = TimeSeriesDiscrete.getIntersectReceptiveFields(bmuSeqs, idx);
												
				Map<double[],BitVector> bvs = new HashMap<double[],BitVector>();
				for( double[] d : rf.keySet() )
					bvs.put( d, TimeSeriesDiscrete.doubleListToBV(rf.get(d)) );
				
				for( double[] d : bvs.keySet() ) {
					log.debug("rf "+bmuSeqs.get(d).size()+": "+TimeSeriesDiscrete.toString(bvs.get(d)));
					for( List<double[]> l : bmuSeqs.get(d).subList(0, 10) ) {
						StringBuffer sb = new StringBuffer();
						for( double[] x : l )
							if(x[idx] > 0.0 )
								sb.append("1");
							else
								sb.append("0");
						//log.debug(sb);
						
					}
				}
				
				{
				Set<BitVector> s = new HashSet<BitVector>(bvs.values());
				log.debug(s.size());
				log.debug(bvs.values().size());
				}
				
				// get depth			
				double depth = 0;
				for( double[] bmu : bmus.keySet() )
					depth +=  (double)(bmus.get(bmu).size() * rf.get(bmu).size())/samples.size();
				log.debug("depth: "+depth);
				
				Set<BitVector> s = new HashSet<BitVector>();
				for( List<Double> l : rf.values() ) {
					List<Double> lr = new ArrayList<Double>(l);
					Collections.reverse(lr);
					s.add( TimeSeriesDiscrete.doubleListToBV(lr));
				}
									
				TimeSeriesDiscrete.saveGnuplotTree( s , "output/tree.dat" );
				
				log.debug("-------------");
			}
		}
			
	}

}

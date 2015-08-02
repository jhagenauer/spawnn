package context.cng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

public class TestWeighted {

	private static Logger log = Logger.getLogger(TestWeighted.class);
	
	public static void main(String[] args) {
		
		double maxDiff = -1;

		// best ratio 0.49, noise 0.7
		for( double ratio : new double[]{0.49} ) {
			for( double noise : new double[]{0.7}) {
		//for (double ratio = 0.01; ratio < 0.5; ratio += 0.01) {
		//	for (double noise = 0.0; noise <= 1; noise += 0.01) {
				Random r = new Random();
			
				List<double[]> samples = new ArrayList<double[]>();
				int c0 = 0, c1 = 0;
											
				for (int i = 0; i < 1000; i++) {
					double x = r.nextDouble();
					double y = r.nextDouble();
					
					if( c0 < 500 && ( x < 0.5 - ratio || x > 0.5 + ratio ) ) { // c0, no noise
						samples.add( new double[]{ x,y,0.0,0 } );
						c0++;
					} else if( c1 < 500 ){ // c1, noisy
						samples.add( new double[]{ x,y,1.0 - noise/2 + r.nextDouble()*noise,1 } );
						c1++;
					}
				}
				
				int classCol = 3;
				int[] fa = { 2 };
				int[] ga = new int[] { 0, 1 };

				// DataUtil.normalizeColumns(samples, fa);

				Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();
				for (double[] d : samples) {
					int c = (int) d[classCol];
					if (!classes.containsKey(c))
						classes.put(c, new HashSet<double[]>());
					classes.get(c).add(d);
				}
				
				// DataUtil.normalizeColumns(samples, fa);
				// DataUtil.normalizeGeoColumns(samples, ga);

				int T_MAX = 100000;
				int numCluster = classes.size();

				Dist<double[]> eDist = new EuclideanDist();
				Dist<double[]> fDist = new EuclideanDist(fa);
				Dist<double[]> gDist = new EuclideanDist(ga);

				// weighted ng
				
				double maxNMIwng = 0;
				{
					for (double w = 0.0; w <= 1; w += 0.1) {

						Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
						m.put(fDist, w);
						m.put(gDist, 1 - w);
						Dist<double[]> wd = new WeightedDist<double[]>(m);

						Sorter bmuGetter = new DefaultSorter(wd);
						NG ng = new NG(numCluster, (double) numCluster / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
						for (double[] p : ng.getNeurons())
							cluster.put(p, new HashSet<double[]>());
						for (double[] d : samples) {
							bmuGetter.sort(d, ng.getNeurons());
							double[] bmu = ng.getNeurons().get(0);
							cluster.get(bmu).add(d);
						}
						double nmi = DataUtils.getNormalizedMutualInformation(cluster.values(), classes.values());
						//log.info(ratio+", basic ng, "+w+", "+nmi);
						if (nmi > maxNMIwng)
							maxNMIwng = nmi;
					}
					//log.info("basic ng, "+ratio+", "+noise+", "+maxNMIwng);
				}

				// cng
				double maxNMIcng = 0;
				{
					for (int sns = 1; sns <= numCluster; sns++) {
						Sorter bmuGetter = new KangasSorter(gDist, fDist, sns);
						NG ng = new NG(numCluster, (double) numCluster / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
						for (double[] p : ng.getNeurons())
							cluster.put(p, new HashSet<double[]>());
						for (double[] d : samples) {
							bmuGetter.sort(d, ng.getNeurons());
							double[] bmu = ng.getNeurons().get(0);
							cluster.get(bmu).add(d);
						}

						double nmi = DataUtils.getNormalizedMutualInformation(cluster.values(), classes.values());
						if (nmi > maxNMIcng)
							maxNMIcng = nmi;
						// log.info(ratio+", cng, "+sns+", "+nmi);
					}
					//log.info("cng, "+ratio+", "+noise+", "+maxNMIwng);
				}
				
				if( maxNMIcng - maxNMIwng > maxDiff ) {
					maxDiff = maxNMIcng - maxNMIwng;
					log.debug("max: "+ratio+", "+noise+", "+maxNMIwng+", "+maxNMIcng+", "+maxDiff);
				}
				
				if( maxNMIcng - maxNMIwng > .5 ) {
					log.debug("above: "+ratio+", "+noise+", "+maxNMIwng+", "+maxNMIcng );
				}
			}
		}
	}
}

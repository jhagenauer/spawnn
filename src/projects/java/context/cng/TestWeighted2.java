package context.cng;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.random.AbstractRandomGenerator;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.ClusterValidation;

public class TestWeighted2 {

	private static Logger log = Logger.getLogger(TestWeighted2.class);

	/* TODO: nur nähe darf nicht zu gut sein. idee: cluster sollten sich überlappen, z.B. gaussche verteilung
	 *       Problem: sns=2 ist nutzlos, sns=1
	 *       ist es ein gutes qualitätskriterium einfach die erfolgreichen zu zählen? dneke nicht.
	 *       Es interessiert vielmehr, welche samples falsch erkannt werden
	 */
	public static void main(String[] args) {
	
		class RndGen extends AbstractRandomGenerator {
			private Random r = new Random();

			@Override
			public double nextDouble() {
				return r.nextDouble();
			}

			@Override
			public void setSeed(long s) {
				clear();
				r.setSeed(s);
			}	
		}
		
		RndGen r = new RndGen();

		List<double[]> samples = new ArrayList<double[]>();

		for (int i = 0; i < 1000; i++) {
			int c = r.nextInt(3);
			
			if( c == 0 ) {
				double x = r.nextGaussian();
				samples.add(new double[] { x, 0, 0 });
			} else if( c == 1 ) {
				double x = r.nextGaussian()+1;
				samples.add(new double[] { x, 1, 1 });
			} else {
				double x = r.nextGaussian()+2;
				samples.add(new double[] { x, 0, 2 });
			}
						
			/*if( c == 0 ) {
				double x = r.nextDouble()-0.5;
				samples.add(new double[] { x, 0, 0 });
			} else if( c == 1 ) {
				double x = r.nextDouble()+1-0.5;
				samples.add(new double[] { x, 1, 1 });
			} else {
				double x = r.nextDouble()+2-0.5;
				samples.add(new double[] { x, 0, 2 });
			}*/
		}
		
		Set<double[]> neurons = new HashSet<double[]>();
		neurons.add( new double[]{0,0,1} );
		neurons.add( new double[]{1,1,1} );
		neurons.add( new double[]{2,0,3} );
					
		int classCol = 2;
		int[] fa = { 1 };
		int[] ga = new int[] { 0 };

		Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();
		for (double[] d : samples) {
			int c = (int) d[classCol];
			if (!classes.containsKey(c))
				classes.put(c, new HashSet<double[]>());
			classes.get(c).add(d);
		}
		
		List<double[]> testSamples = new ArrayList<double[]>();
		for (double noise = 0; noise <= 1; noise += 0.1) 
			for (double d = 0; d <= 1; d += 0.1) 
				testSamples.add( new double[] { d, noise, 0 } );

		int T_MAX = 100000;
		int numCluster = classes.size();
		int numNeurons = numCluster;

		Dist<double[]> eDist = new EuclideanDist();
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		double bestNmi = 0;
		Set<double[]> wngCorrect = new HashSet<double[]>();
		for (double w = 0.0; w <= 1; w += 0.1) {
			Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
			m.put(fDist, w);
			m.put(gDist, 1 - w);
			Dist<double[]> wd = new WeightedDist<double[]>(m);

			Sorter bmuGetter = new DefaultSorter(wd);
			NG ng = new NG(numNeurons, (double) numNeurons / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

			for (int t = 0; t < T_MAX; t++)
				ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
						
			Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
			for (double[] p : ng.getNeurons())
				cluster.put(p, new HashSet<double[]>());
			for (double[] d : samples) {
				bmuGetter.sort(d, ng.getNeurons());
				double[] bmu = ng.getNeurons().get(0);
				cluster.get(bmu).add(d);
			}
			double nmi = ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values());
			if( nmi > bestNmi )
				bestNmi = nmi;
			
			double[] class0Neuron = null;
			int best = 0;
			for( double[] n : cluster.keySet() ) {
				int count0 = 0;
				for( double[] d : cluster.get(n) )
					if( d[classCol] == 0 )
						count0++;
				if( count0 > best ) {
					best = count0;
					class0Neuron = n;
				}
			}
							
			for( double[] x : testSamples ) {
				double[] bmu = ng.getNeurons().get(0);
				bmuGetter.sort(x, ng.getNeurons());
				if (bmu == class0Neuron )
					wngCorrect.add(x);
			}
		}
		log.debug("best nmi wng: "+bestNmi);
				
		List<double[]> notCorrect = new ArrayList<double[]>(testSamples);
		notCorrect.removeAll(wngCorrect);
		log.debug("wng did correctly assign "+wngCorrect.size()+" samples");
		log.debug("wng did not correctly assign "+notCorrect.size()+" samples");

		Set<double[]> cngCorrect = new HashSet<double[]>();
		bestNmi = 0;
		for (int sns = 1; sns <= numNeurons; sns++) {
			Sorter bmuGetter = new KangasSorter(gDist, fDist, sns);
			NG ng = new NG(numNeurons, (double) numNeurons / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

			for (int t = 0; t < T_MAX; t++)
				ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
			
			Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
			for (double[] p : ng.getNeurons())
				cluster.put(p, new HashSet<double[]>());
			for (double[] d : samples) {
				bmuGetter.sort(d, ng.getNeurons());
				double[] bmu = ng.getNeurons().get(0);
				cluster.get(bmu).add(d);
			}
			double nmi = ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values());
			if( nmi > bestNmi )
				bestNmi = nmi;
			
			double[] class0Neuron = null;
			int best = 0;
			for( double[] n : cluster.keySet() ) {
				int count0 = 0;
				for( double[] d : cluster.get(n) )
					if( d[classCol] == 0 )
						count0++;
				if( count0 > best ) {
					best = count0;
					class0Neuron = n;
				}
			}

			for( double[] x : testSamples ) {
				bmuGetter.sort(x, ng.getNeurons());
				double[] bmu = ng.getNeurons().get(0);
				if (bmu == class0Neuron ) {
					cngCorrect.add(x);
				}
			}
		}
		log.debug("best nmi cng: "+bestNmi);
		
		List<double[]> l = new ArrayList<double[]>(cngCorrect);
		l.removeAll(wngCorrect);
		log.debug("cng correctly detected: ");
		for( double[] d : l )
			log.debug(Arrays.toString(d));
	}
}

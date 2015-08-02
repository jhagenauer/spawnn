package context.cng;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;

public class TestWeighted3 {

	private static Logger log = Logger.getLogger(TestWeighted3.class);

	/* TODO: nur nähe darf nicht zu gut sein. idee: cluster sollten sich überlappen, z.B. gaussche verteilung
	 *       Problem: sns=2 ist nutzlos, sns=1
	 *       ist es ein gutes qualitätskriterium einfach die erfolgreichen zu zählen? dneke nicht.
	 *       Es interessiert vielmehr, welche samples falsch erkannt werden
	 */
	public static void main(String[] args) {
		int[] fa = { 1 };
		int[] ga = new int[] { 0 };
		
		List<double[]> testSamples = new ArrayList<double[]>();
		for (double noise = 0.5; noise <= 2.5; noise += 0.01) 
			testSamples.add( new double[] { 0.5, noise, 0 } );

		int numCluster = 3;
		int numNeurons = numCluster;

		Dist<double[]> eDist = new EuclideanDist();
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);
		
		log.debug("wng:");
		{
			Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
			m.put(fDist, 0.5);
			m.put(gDist, 1 - 0.5);
			Dist<double[]> wd = new WeightedDist<double[]>(m);

			Sorter bmuGetter = new DefaultSorter(wd);
			
			Set<double[]> neurons = new HashSet<double[]>();
			neurons.add( new double[]{0.5,0,0} );
			neurons.add( new double[]{1.5,1,1} );
			neurons.add( new double[]{2.5,2,2} );
			NG ng = new NG( neurons, (double) numNeurons / 2, 0.01, 0.5, 0.005, bmuGetter);
			
			
			log.debug("neurons: ");
			for(double[] n : ng.getNeurons() )
				log.debug(Arrays.toString(n));
			
			log.debug("bmus:");
			for( double[] x : testSamples ) {
				bmuGetter.sort(x, ng.getNeurons() );
				log.debug(x[1] +" -> "+ng.getNeurons().get(0)[1] );
			}
				
		}
		
		log.debug("cng: ");
		{
			Sorter bmuGetter = new KangasSorter(gDist, fDist, 2);
			Set<double[]> neurons = new HashSet<double[]>();
			neurons.add( new double[]{0.5,0,0} );
			neurons.add( new double[]{1.5,1,1} );
			neurons.add( new double[]{2.5,2,2} );
			NG ng = new NG( neurons, (double) numNeurons / 2, 0.01, 0.5, 0.005, bmuGetter);
			
			log.debug("neurons: ");
			for(double[] n : ng.getNeurons() )
				log.debug(Arrays.toString(n));
			
			log.debug("bmus:");
			for( double[] x : testSamples ) {
				bmuGetter.sort(x, ng.getNeurons() );
				log.debug(x[1] +" -> "+ng.getNeurons().get(0)[1] );
			}
		}
	}
}

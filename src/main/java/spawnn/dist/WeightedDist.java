package spawnn.dist;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;

import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;

public class WeightedDist<T> implements Dist<T> {

	private Map<Dist<T>, Double> dists;
	
	private List<Dist<T>> l1;
	private double[] l2;

	public WeightedDist(Map<Dist<T>, Double> dists) {
		this.dists = dists;
		
		l1 = new ArrayList<Dist<T>>(dists.keySet());
		l2 = new double[l1.size()];
		for( int i = 0; i < l1.size(); i++ )
			l2[i] = dists.get( l1.get(i) );
	}

	@Override
	public double dist(T a, T b) {
		double sum = 0;
		//FIXME This is awfully slow. Don't know why... ArrayLists as workaround
		/*for(Entry<Dist<T>, Double> e : dists.entrySet() )
			sum += e.getValue() * e.getKey().dist(a, b);*/
		for( int i = 0; i < l1.size(); i++ )
			sum += l1.get(i).dist(a, b) * l2[i];
		return sum;
	}

	public static void main(String[] args) {
		JDKRandomGenerator r = new JDKRandomGenerator();
		GaussianRandomGenerator grg = new GaussianRandomGenerator(r);

		final int nrSamples = 10000;

		int nrGDim = 2;
		int nrFDim = 5;
		final List<double[]> samples = new ArrayList<double[]>();
		for (int i = 0; i < nrSamples; i++) {
			double[] d = new double[nrGDim + nrFDim];
			for (int j = 0; j < nrGDim; j++)
				d[j] = grg.nextNormalizedDouble();

			for (int j = 0; j < nrFDim; j++)
				d[j + nrGDim] = grg.nextNormalizedDouble();
			samples.add(d);
		}

		final int[] ga = new int[nrGDim];
		for (int i = 0; i < nrGDim; i++)
			ga[i] = i;
		final int[] fa = new int[nrFDim];
		for (int i = 0; i < nrFDim; i++)
			fa[i] = i + nrGDim;

		Dist<double[]> gDist = new EuclideanDist(ga);
		Dist<double[]> fDist = new EuclideanDist(fa);

		int t_max = 100000;

		Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
		map.put(fDist, 1 - 0.5);
		map.put(gDist, 0.5);

		Dist<double[]> wDist = new WeightedDist<double[]>(map);

		Sorter<double[]> s = new DefaultSorter<double[]>(wDist);
		//s = new KangasSorter<>(gDist, fDist, 25);

		long sum = 0;
		for (int i = 0; i < 100; i++) {

			NG ng = new NG(25, 10, 0.01, 0.5, 0.005, samples.get(0).length, s);

			long time = System.currentTimeMillis();
			for (int t = 0; t < t_max; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train((double) t / t_max, x);
			}
			sum += (System.currentTimeMillis() - time);
		}
		System.out.println((double) sum / 100); // 1078
	}

}

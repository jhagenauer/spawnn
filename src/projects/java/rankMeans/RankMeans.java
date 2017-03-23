package rankMeans;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.Clustering;

public class RankMeans {
	
	public static Random r = new Random();
	
	public static Map<double[], Set<double[]>> kMeans(List<double[]> samples, int num, Sorter<double[]> sorter, double delta ) {
		int length = samples.iterator().next().length;
		Dist<double[]> dist = new EuclideanDist();
			
		Map<double[], Set<double[]>> clusters = null;
		Set<double[]> centroids = new HashSet<double[]>();

		// get num unique(!) indices for centroids
		Set<Integer> indices = new HashSet<Integer>();
		while (indices.size() < num)
			indices.add(r.nextInt(samples.size()));

		for (int i : indices) {
			double[] d = samples.get(i);
			centroids.add(Arrays.copyOf(d, d.length));
		}

		while( true) {
			clusters = new HashMap<double[], Set<double[]>>();
			for (double[] v : centroids)
				// init cluster
				clusters.put(v, new HashSet<double[]>());

			for (double[] s : samples) { // build cluster
				List<double[]> l = new ArrayList<double[]>(clusters.keySet());
				sorter.sort(s, l);
				clusters.get( l.get(0) ).add(s);
			}

			boolean changed = false;
			for (double[] c : clusters.keySet()) {
				Collection<double[]> s = clusters.get(c);

				if (s.isEmpty())
					continue;

				// calculate new centroids
				double[] centroid = new double[length];
				for (double[] v : s)
					for (int i = 0; i < v.length; i++)
						centroid[i] += v[i];
				for (int i = 0; i < centroid.length; i++)
					centroid[i] /= s.size();

				// update centroids
				centroids.remove(c);
				centroids.add(centroid);
				
				if( !changed && dist.dist(c, centroid) > delta )
					changed = true;				
			}			
			if( !changed )
				break;
		} 
		return clusters;
	}

	public static void main(String[] args) {
		double delta = 0.000001;
		
		List<double[]> samples = new ArrayList<double[]>();
		while( samples.size() < 1000 ) 
			samples.add( new double[]{ Math.pow(r.nextDouble(), 2),r.nextDouble() } );
		
		Dist<double[]> aDist = new EuclideanDist( new int[]{0} );
		Dist<double[]> bDist = new EuclideanDist( new int[]{1} );
		Sorter<double[]> aSorter = new DefaultSorter<>( aDist );	
		Sorter<double[]> bSorter = new DefaultSorter<>( bDist );
		
		int numProto = 10;
		
		Path file = Paths.get("output/rankMeans.csv");
		try {
			Files.createDirectories(file.getParent()); // create output dir
			Files.deleteIfExists(file);
			Files.createFile(file);
			String s = "method,k";
			for( int i = 0; i < numProto; i++ )
				s += ",proto_"+i;
			s+= "\r\n";
			Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		
		for( int run = 0; run < 100; run++ ) {
			
			// kangas
			for( int k = 1; k <= 10; k++ ) {
				Map<double[],Set<double[]>> cluster = kMeans(samples, numProto, new KangasSorter<>(aSorter, bSorter, k), delta);
				List<double[]> l = new ArrayList<double[]>(cluster.keySet());
				aSorter.sort( new double[]{ 0,0 }, l);
				
				try {
					String s = "";
					s += "kangas,"+k;
					for( double[] d : l )
						s += ","+d[0];
					s += "\r\n";
					Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
			for( int k = 0; k <= 100; k++ ) {
				
				double p = (double)k/100; 
				Map<Dist<double[]>,Double> m = new HashMap<>();
				m.put(aDist,p);
				m.put(bDist, 1.0-p);
				
				Map<double[],Set<double[]>> cluster = Clustering.kMeans(samples, numProto, new WeightedDist<>(m), delta );
				List<double[]> l = new ArrayList<double[]>(cluster.keySet());
				aSorter.sort( new double[]{ 0,0 }, l);
				
				try {
					String s = "";
					s += "weighted,"+k;
					for( double[] d : l )
						s += ","+d[0];
					s += "\r\n";
					Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}	
	}
}

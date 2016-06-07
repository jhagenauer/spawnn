package regionalization;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.RandomDist;
import spawnn.utils.Drawer;
import spawnn.utils.GraphUtils;

public class MSTRegioTest {
	private static Logger log = Logger.getLogger(MSTRegioTest.class);
	
	public static void main(String[] args ) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while( samples.size() < 25 ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z = r.nextDouble();
			Coordinate c = new Coordinate(x, y);
			coords.add(c);
			geoms.add( gf.createPoint(c));
			samples.add( new double[]{ x,y,z } );
		}
		
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setClipEnvelope(new Envelope(0, 0, 1, 1) );
		vdb.setSites(coords);
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> voroGeoms = new ArrayList<Geometry>();
		for (int i = 0; i < coords.size(); i++) {
			Geometry p = gf.createPoint(coords.get(i));
			for (int j = 0; j < coll.getNumGeometries(); j++) 
				if (p.intersects(coll.getGeometryN(j))) {
					voroGeoms.add(coll.getGeometryN(j));
					break;
				}
		}
		
		// build cm map based on voro
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (int i = 0; i < samples.size(); i++) {
			Set<double[]> s = new HashSet<double[]>();
			for (int j = 0; j < samples.size(); j++) 
				if (i != j && voroGeoms.get(i).intersects(voroGeoms.get(j))) 
					s.add(samples.get(j));
			cm.put(samples.get(i), s);
		}
		
		Dist<double[]> fDist = new EuclideanDist(new int[]{2});
		Dist<double[]> rDist = new RandomDist<double[]>();
		Map<double[], Set<double[]>>  mst = GraphUtils.getMinimumSpanningTree(cm, fDist);
		Map<double[], Set<double[]>>  treeA = GraphUtils.getMinimumSpanningTree(cm, rDist);
		Map<double[], Set<double[]>>  treeB = GraphUtils.getMinimumSpanningTree(cm, rDist);
		
		Drawer.geoDrawConnections(treeA, null, new int[]{0,1},null, "output/treeA.png");
		Drawer.geoDrawConnections(treeB, null, new int[]{0,1},null, "output/treeB.png");
		
		// get random edge 
		boolean even = true;
		boolean aFull = false, bFull = false;
		Set<double[]> addedA = new HashSet<double[]>();
		addedA.add(new ArrayList<double[]>(treeA.keySet()).get(r.nextInt(treeA.keySet().size())));
		Set<double[]> addedB = new HashSet<double[]>();
		addedB.add(new ArrayList<double[]>(treeB.keySet()).get(r.nextInt(treeB.keySet().size())));
		
		Map<double[], Set<double[]>>  nTree = new HashMap<double[],Set<double[]>>();
		// add nodes that belong exclusively to the subtrees
		while( !aFull || !bFull ) {
			log.debug(even+","+aFull+","+bFull+","+addedA.size()+","+addedB.size());
			
			if( even && !aFull ) { // from treeA
				Map<double[],Set<double[]>> c = new HashMap<double[],Set<double[]>>();
				for( double[] a : addedA ) {
					Set<double[]> s = new HashSet<double[]>();
					for( double[] b : treeA.get(a) )
						if( !addedA.contains(b) && !addedB.contains(b) )
							s.add(b);
					if( !s.isEmpty() )
						c.put(a, s);
				}
				if( !c.isEmpty() ) {
					double[] na = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
					double[] nb = new ArrayList<double[]>(c.get(na)).get(r.nextInt(c.get(na).size()));
					
					// add connections to both directions
					if (!nTree.containsKey(na))
						nTree.put(na, new HashSet<double[]>());
					nTree.get(na).add(nb);
					if (!nTree.containsKey(nb))
						nTree.put(nb, new HashSet<double[]>());
					nTree.get(nb).add(na);
					addedA.add(nb);
				} else 
					aFull = true;
			} else if( !bFull ){ // from treeB
				Map<double[],Set<double[]>> c = new HashMap<double[],Set<double[]>>();
				for( double[] a : addedB ) {
					Set<double[]> s = new HashSet<double[]>();
					for( double[] b : treeB.get(a) )
						if( !addedA.contains(b) && !addedB.contains(b) )
							s.add(b);
					if( !s.isEmpty() )
						c.put(a, s);
				}
				if( !c.isEmpty() ) {
					double[] na = new ArrayList<double[]>(c.keySet()).get(r.nextInt(c.keySet().size()));
					double[] nb = new ArrayList<double[]>(c.get(na)).get(r.nextInt(c.get(na).size()));
					
					// add connections to both directions
					if (!nTree.containsKey(na))
						nTree.put(na, new HashSet<double[]>());
					nTree.get(na).add(nb);
					if (!nTree.containsKey(nb))
						nTree.put(nb, new HashSet<double[]>());
					nTree.get(nb).add(na);
					addedB.add(nb); 
				} else
					bFull = true;
			}
			even = !even;
		}
		
		Drawer.geoDrawConnections(nTree, null, new int[]{0,1},null, "output/nTree_nocon.png");
		
		// connect subtrees	
		{
			Map<double[],Set<double[]>> conn = new HashMap<double[],Set<double[]>>();
			for( double[] a : addedA ) {
				Set<double[]> s = new HashSet<double[]>();
				for( double[] b : addedB )
					if( treeA.get(a).contains(b) )
						s.add(b);
				if( !s.isEmpty() )
					conn.put(a, s);
			}
			
			double[] na = new ArrayList<double[]>(conn.keySet()).get(r.nextInt(conn.keySet().size()));
			double[] nb = new ArrayList<double[]>(conn.get(na)).get(r.nextInt(conn.get(na).size()));
			
			if( !nTree.containsKey(na) )
				nTree.put(na,new HashSet<double[]>() );
			nTree.get(na).add(nb);
			if( !nTree.containsKey(nb) )
				nTree.put(nb,new HashSet<double[]>() );
			nTree.get(nb).add(na);
			
			Map<double[],double[]> hl = new HashMap<double[],double[]>();
			hl.put(na, nb);
			
			Drawer.geoDrawConnections(nTree, null, new int[]{0,1},null, "output/nTree_con.png");
		}
				
		// build complete spanning tree, prefer edges from treeA and treeB
		Set<double[]> added = new HashSet<double[]>(addedA);
		added.addAll(addedB);
		while( added.size() != samples.size() ) {
			
			Map<double[],Set<double[]>> fromTrees = new HashMap<double[],Set<double[]>>();
			Map<double[],Set<double[]>> nEdges = new HashMap<double[],Set<double[]>>();
			for (double[] a : added) {
				Set<double[]> s1 = new HashSet<double[]>();
				for (double[] b : treeA.get(a))
					if ( !added.contains(b) )
						s1.add(b);
				for (double[] b : treeB.get(a))
					if ( !added.contains(b) )
						s1.add(b);
				if( !s1.isEmpty() )
					fromTrees.put(a,s1);
				
				Set<double[]> s2 = new HashSet<double[]>();
				for( double[] b : cm.get(a) )
					if( !added.contains(b) && s1.contains(b) )
						s2.add(b);
				if( !s2.isEmpty() )
					nEdges.put(a, s2);		
			}
			
			double[] na;
			double[] nb;
			if( !fromTrees.isEmpty() ) {
				na = new ArrayList<double[]>(fromTrees.keySet()).get(r.nextInt(fromTrees.keySet().size()));
				nb = new ArrayList<double[]>(fromTrees.get(na)).get(r.nextInt(fromTrees.get(na).size()));
			} else { // nEdges should never be empty
				na = new ArrayList<double[]>(nEdges.keySet()).get(r.nextInt(nEdges.keySet().size()));
				nb = new ArrayList<double[]>(nEdges.get(na)).get(r.nextInt(nEdges.get(na).size()));
			}
			
			// add connections to both directions
			if (!nTree.containsKey(na))
				nTree.put(na, new HashSet<double[]>());
			nTree.get(na).add(nb);

			if (!nTree.containsKey(nb))
				nTree.put(nb, new HashSet<double[]>());
			nTree.get(nb).add(na);

			added.add(nb);	
		}		
		Drawer.geoDrawConnections(nTree, null, new int[]{0,1},null, "output/nTree_full.png");
		
		// cuts
		Map<double[],Set<double[]>> cuts = new HashMap<double[],Set<double[]>>();
		int numCuts = 0;
		while( numCuts < 4 ) {
			double[] na = new ArrayList<double[]>(nTree.keySet()).get(r.nextInt(nTree.keySet().size()));
			double[] nb = new ArrayList<double[]>(nTree.get(na)).get(r.nextInt(nTree.get(na).size()));
			
			if( !cuts.containsKey(na) || !cuts.get(na).contains(nb) ) {
				if( !cuts.containsKey(na) )
					cuts.put(na, new HashSet<double[]>() );
				cuts.get(na).add(nb);
				
				if( !cuts.containsKey(nb) )
					cuts.put(nb, new HashSet<double[]>() );
				cuts.get(nb).add(na);
				numCuts++;
			}
		}
		Drawer.geoDrawConnections(cuts, null, new int[]{0,1},null, "output/cuts.png");
		
		// to cluster
		Map<double[],Set<double[]>> cutTree = new HashMap<double[],Set<double[]>>();
		for( Entry<double[],Set<double[]>> e : nTree.entrySet() ) {
			Set<double[]> s = new HashSet<double[]>(e.getValue());
			if( cuts.containsKey(e.getKey() ))
				s.removeAll(cuts.get(e.getKey()));
			cutTree.put(e.getKey(), s);
		}
		Drawer.geoDrawConnections(cutTree, null, new int[]{0,1},null, "output/cutTree.png");
		
		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for (Map<double[], Set<double[]>> sg : GraphUtils.getSubGraphs(cutTree)) {
			Set<double[]> nodes = new HashSet<double[]>(sg.keySet());
			for (double[] a : sg.keySet())
				nodes.addAll(sg.get(a));
			clusters.add(nodes);
		}
		
		Drawer.geoDrawCluster(clusters, samples, voroGeoms, "output/cluster.png", true);
		Drawer.geoDrawCluster(clusters, samples, geoms, "output/cluster_points.png", true);
	}
}

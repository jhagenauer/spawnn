package regionalization;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.Style;
import org.geotools.styling.StyleBuilder;
import org.geotools.styling.TextSymbolizer;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class MSTRegio {
	private static Logger log = Logger.getLogger(MSTRegio.class);
	
	public static void main(String[] args ) {
		GeometryFactory gf = new GeometryFactory();
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();
		List<Coordinate> coords = new ArrayList<Coordinate>();
		while( samples.size() < 20 ) {
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
		Map<double[], Set<double[]>>  treeA = getSpanningTree(cm, fDist);
		Map<double[], Set<double[]>>  treeB = getSpanningTree(cm, fDist);
		
		// get intersection 
		Map<double[],Set<double[]>> is = new HashMap<double[],Set<double[]>>();
		for( double[] a : treeA.keySet() ) {
			Set<double[]> s = new HashSet<double[]>();
			for( double[] b : treeA.get(a) )
				if( treeB.containsKey(a) && treeB.get(a).contains(b) )
					s.add(b);
			if( !s.isEmpty() )
				is.put(a,s);
		}
		// get random edge from intersection
		double[] ra = new ArrayList<double[]>(is.keySet()).get(r.nextInt(is.keySet().size()));
		double[] rb = new ArrayList<double[]>(is.get(ra)).get(r.nextInt(is.get(ra).size()));
		
		Map<double[],double[]> hl = new HashMap<double[],double[]>();
		hl.put(ra, rb);
		
		geoDrawSpanningTree(treeA, hl, new int[]{0,1},null, "output/treeA.png");
		geoDrawSpanningTree(treeB, hl, new int[]{0,1},null, "output/treeB.png");
		geoDrawSpanningTree(is, hl, new int[]{0,1},null, "output/intersection.png");
				
		// combine mstA and mstB to new mst
		treeA.get(ra).remove(rb);
		treeB.get(rb).remove(ra);
		
		boolean even = true;
		boolean aFull = false, bFull = false;
		Set<double[]> addedA = new HashSet<double[]>();
		addedA.add(ra);
		Set<double[]> addedB = new HashSet<double[]>();
		addedB.add(rb);
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
				
		// repair orig trees
		treeA.get(ra).add(rb);
		treeB.get(rb).add(ra);
		
		// merge subtrees, new sets are necessary if subtree is only one node
		if( !nTree.containsKey(ra) )
			nTree.put(ra,new HashSet<double[]>() );
		nTree.get(ra).add(rb);
		if( !nTree.containsKey(rb) )
			nTree.put(rb,new HashSet<double[]>() );
		nTree.get(rb).add(ra);
		
		geoDrawSpanningTree(nTree, hl, new int[]{0,1},null, "output/nTree.png");
		
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
		geoDrawSpanningTree(nTree, hl, new int[]{0,1},null, "output/nTree_full.png");
	}
	
	public static Map<double[], Set<double[]>> getSpanningTree(Map<double[], Set<double[]>> cm, Dist<double[]> dist) {
		Map<double[], Set<double[]>> tree = new HashMap<double[], Set<double[]>>();
		
		Set<double[]> s = new HashSet<double[]>(cm.keySet());
		for( Set<double[]> sub : cm.values() )
			s.addAll(sub);
		List<double[]> l = new ArrayList<double[]>(s);
		Random r = new Random();

		Set<double[]> added = new HashSet<double[]>();
		added.add(l.get(r.nextInt(l.size())));

		while (added.size() != cm.size()) { 

			double[] bestA = null, bestB = null;
			double minDist = Double.MAX_VALUE;
			for (double[] a : added) {
				for (double[] b : cm.get(a)) {
					if (added.contains(b))
						continue;

					//double d = dist.dist(a, b);
					double d = r.nextDouble();
					if (d < minDist) {
						minDist = d;
						bestA = a;
						bestB = b;
					}
				}
			}

			// add connections to both directions
			if (!tree.containsKey(bestA))
				tree.put(bestA, new HashSet<double[]>());
			tree.get(bestA).add(bestB);

			if (!tree.containsKey(bestB))
				tree.put(bestB, new HashSet<double[]>());
			tree.get(bestB).add(bestA);

			added.add(bestB);
		}
		return tree;
	}
	
	public static void geoDrawSpanningTree(Map<double[], Set<double[]>> mst, Map<double[],double[]> hl, int[] ga, CoordinateReferenceSystem crs, String fn) {
		GeometryFactory gf = new GeometryFactory();
		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			ReferencedEnvelope mapBounds = mc.getMaxBounds();

			// lines
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("lines");
				typeBuilder.setCRS(crs);
				typeBuilder.add("the_geom", LineString.class);

				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				for (double[] a : mst.keySet())
					for (double[] b : mst.get(a) ) {
						//featureBuilder.set("age", e.getValue());
						LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
						featureBuilder.set("the_geom", ls);
						features.add(featureBuilder.buildFeature("" + features.size()));
					}

				Style style = SLD.wrapSymbolizers(sb.createLineSymbolizer(2.0));
				mc.addLayer(new FeatureLayer(features, style));
				
				// highlights
				DefaultFeatureCollection hlFeatures = new DefaultFeatureCollection();
				
				if( hl != null ) {
					for(Entry<double[],double[]> e : hl.entrySet() ) {
						LineString ls = gf.createLineString(new Coordinate[] { new Coordinate(e.getKey()[ga[0]], e.getKey()[ga[1]]), new Coordinate(e.getValue()[ga[0]], e.getValue()[ga[1]]) });
						featureBuilder.set("the_geom", ls);
						hlFeatures.add(featureBuilder.buildFeature("" + features.size()));
					}
					mc.addLayer(new FeatureLayer(hlFeatures, SLD.wrapSymbolizers(sb.createLineSymbolizer(Color.RED,2.0))));
				}
			}
			
			// points
			{
				SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
				typeBuilder.setName("points");
				typeBuilder.setCRS(crs);
				typeBuilder.add("the_geom", Point.class);

				SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
				DefaultFeatureCollection features = new DefaultFeatureCollection();
				Set<double[]> s = new HashSet<double[]>(mst.keySet());
				for( Set<double[]> m : mst.values() )
					s.addAll(m);
				for (double[] a :s ) {
					Point p = gf.createPoint( new Coordinate( a[ga[0]], a[ga[1]] ) );
					featureBuilder.set("the_geom", p);
					features.add(featureBuilder.buildFeature("" + features.size()));
				}
				
				Style style = SLD.wrapSymbolizers(sb.createPointSymbolizer());
				FeatureLayer fl = new FeatureLayer(features, style);
				mc.addLayer(fl);
				mapBounds.expandToInclude(fl.getBounds());
			}

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);
		
			Rectangle imageBounds = null;

			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 1000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", new FileOutputStream(fn));
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

// point of this program:
// starting with data like the Nexrad radial velocity (r, theta, velocity toward/away from origin)
// example: http://www.intellicast.com/National/Nexrad/RadialVelocity.aspx?location=USKY1447
// construct a raw vector field equal to the data and attempt to extrapolate from it into a field at all points in the max radius
// use some radial basis function that vanishes at finite distance
// be able to approximate the curl at a point (just use a small radius around it)
// use curl to find tornadoes


import java.lang.Math;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


class Point {
    public static final int CARTESIAN = 0;
    public static final int POLAR = 1;

    private double[] xy = new double[2];
    private double[] rt = new double[2];

    public Point(double[] coords, int coordinateType) {
        switch(coordinateType) {
            case CARTESIAN:
                this.xy = coords;
                this.rt = cartesianToPolar(coords);
                break;
            case POLAR:
                this.xy = polarToCartesian(coords);
                this.rt = coords;
                break;
            default:
                throw new RuntimeException("invalid option for coordinate type; use Point.CARTESIAN or Point.POLAR");
        }
    }

    public double getX() { return this.xy[0]; }
    public double getY() { return this.xy[1]; }
    public double getR() { return this.rt[0]; }
    public double getTheta() { return this.rt[1]; }

    public static double[] cartesianToPolar(double[] xy) {
        verifyCoords(xy);
        double r = Math.sqrt(Math.pow(xy[0], 2) + Math.pow(xy[1], 2));
        double theta = Math.atan(xy[1] / xy[0]);

        return new double[] {r, theta};
    }

    public static double[] polarToCartesian(double[] rt) {
        verifyCoords(rt);
        double x = rt[0] * Math.cos(rt[1]);
        double y = rt[0] * Math.sin(rt[1]);

        return new double[] {x, y};
    }

    public static void verifyCoords(double[] coords) {
        if (coords.length != 2) {
            throw new RuntimeException("coordinate array must have length 2");
        }
    }

    public boolean isVector() {
        return false;
    }

    public String toString() {
        String typeString = this.isVector() ? "Vector" : "Point";
        return String.format("%s: Cartesian %s; Polar %s", typeString, Arrays.toString(this.xy), Arrays.toString(this.rt));
    }
}


class Vector extends Point {
    public Vector(double[] coords, int coordinateType) {
        super(coords, coordinateType);
    }

    public boolean isVector() {
        return true;
    }

    public double getMagnitude() {
        return this.getR();
    }

    public Vector scaleToMagnitude(double mag) {
        if (mag < 0) {
            throw new RuntimeException("cannot scale vector to negative magnitude");
        }
        double currentMagnitude = this.getMagnitude();
        if (currentMagnitude == 0 && mag != 0) {
            throw new RuntimeException("cannot scale zero vector to nonzero magnitude");
        }
        double factor = mag / currentMagnitude;
        double[] components = new double[] {this.getX() * factor, this.getY() * factor};
        return new Vector(components, Point.CARTESIAN);
    }
}


class VectorFieldMap {
    private Map<Point, Vector> map;

    public VectorFieldMap() {
        this.map = new HashMap<Point, Vector>();
    }
    
    public VectorFieldMap(Map<Point, Vector> map) {
        this.map = map;
    }

    public void put(Point point, Vector vector) {
        if (this.map.containsKey(point)) {
            throw new RuntimeException(String.format("VectorFieldMap already contains point %s", point.toString()));
        }
        this.map.put(point, vector);
    }
}


class VectorField {
    // for now, just implement using VectorFieldMap and kernel for interpolation
    // later, can try to pass functions that generate the components directly

    private VectorFieldMap vectorFieldMap;

    public VectorField(VectorFieldMap vectorFieldMap) {
        this.vectorFieldMap = vectorFieldMap;
    }

    public Vector getVectorAtPoint(Point p) {
        if (this.vectorFieldMap.containsKey(p)) {
            return vectorFieldMap.get(p);
        }

        // currently using super hacky interpolation
        // for better methods, see:
        // - http://www.spc.noaa.gov/publications/schaefer/interpol.pdf
        // - http://www.smpp.northwestern.edu/~smpp_pub/MussaIvaldiBiolCyb1992.pdf

        ;
    }

    public double getCurlAtPoint(Point p) {
        ;
    }

    private Point[] getNearestDataPointsToPoint(Point p, int nPoints) {
        if (nPoints > this.vectorFieldMap.size()) {
            ;
        }
        ;
    }


}


class RadialVectorField extends VectorField {
    // representation of Nexrad-style radial velocity data; really a scalar field with an implicit radial unit vector at each point
    // kept as an array of points with vectors

    private static final Point DEFAULT_CENTER = new Point(new double[] {0, 0}, Point.CARTESIAN);
    private static final double DEFAULT_RADIUS_RESOLUTION = 1.0;

    public RadialVectorField(VectorFieldMap vectorFieldMap) {
        this.vectorFieldMap = vectorFieldMap;
    }

    public static RadialVectorField fromArray(double[][] a) {
        // rows in order of increasing radius (no measurement at center itself)
        // columns iterating from standard position (first column is in the positive x direction from the origin), counterclockwise

        if (a.length == 0 || a[0].length == 0) {
            throw new RuntimeException("array must be nonempty in both dimensions");
        }

        final int nRows = a.length;
        final int nCols = a[0].length;

        final Point center = DEFAULT_CENTER;
        final double radiusResolution = DEFAULT_RADIUS_RESOLUTION;
        final double thetaResolutionDegrees = 360.0 / a[0].length;

        VectorFieldMap vectorFieldMap = new VectorFieldMap();

        for (int i = 0; i < nRows; i++) {
            if (a[i].length != nCols) {
                throw new RuntimeException("cannot create RadialVectorField from array with differing row lengths");
            }
            for (int j = 0; j < nCols; j++) {
                double mag = a[i][j];
                double[] rt = new double[] {(i + 1) * radiusResolution, j * thetaResolutionDegrees};
                Point p = new Point(rt, Point.POLAR);
                Vector radialUnitVector = getRadialUnitVector(center, p);
                Vector radialVector = radialUnitVector.scaleToMagnitude(mag);
                vectorFieldMap.put(p, radialVector);
            }
        }

        return new RadialVectorField(vectorFieldMap);
    }

    public static Vector getRadialUnitVector(Point center, Point towardPoint) {
        double[] coords = new double[] {towardPoint.getX() - center.getX(), towardPoint.getY() - center.getY()};
        Vector fullVector = new Vector(coords, Point.CARTESIAN);

        return fullVector.scaleToMagnitude(1);
    }
}


class TornadoField {
    public static void main(String[] args) {
        double[][] nexradArray = new double[][] {
            {  3.2,  2.7,  2.2,  1.4,  0.8},
            {  2.8,  2.4,  1.6,  0.7,  0.1},
            {  2.0,  1.3,  0.7, -0.1, -0.6},
            {  1.0,  0.2, -0.4, -1.3, -1.8},
            { -0.3, -1.5, -1.8, -2.9, -3.5},
        };

        RadialVectorField radialVectorField = RadialVectorField.fromArray(nexradArray);

        // TODO:
        // - get VectorField (extrapolate using a simple rbf)
        // - find points with high curl
        // - learn how to plot in Java
        //   - plot color-coded magnitude of points
        //   - plot vectors as arrows
        //   - plot curl
        // - be able to handle null for points with no data
    }
}
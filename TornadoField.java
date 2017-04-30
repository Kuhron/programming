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

    double[] xy = new double[2];
    double[] rt = new double[2];

    public Point(double[] coords, int coordinateType) {
        switch(coordinateType) {
            case CARTESIAN:
                this.xy = coords;
                this.rt = cartesianToPolar(coords);
            case POLAR:
                this.xy = polarToCartesian(coords);
                this.rt = coords;
            default:
                throw new RuntimeException("invalid option for coordinate type; use Point.CARTESIAN or Point.POLAR");
        }
    }

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
}


class VectorField {

}


class VectorFieldMap {
    Map<Point, Vector> map;

    public VectorFieldMap() {
        this.map = new HashMap<Point, Vector>();  // just picked HashMap because whatever
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


class RadialVectorField {
    // representation of Nexrad-style radial velocity data; really a scalar field with an implicit radial unit vector at each point
    // kept as an array of points with vectors

    static double[] DEFAULT_CENTER = new double[] {0, 0};

    VectorFieldMap vectorFieldMap;

    public RadialVectorField(VectorFieldMap vectorFieldMap) {
        this.vectorFieldMap = vectorFieldMap;
    }

    public static RadialVectorField fromArray(double[][] a) {
        // rows in order of increasing radius (no measurement at center itself)
        // columns iterating from standard position (first column is in the positive x direction from the origin), counterclockwise

        if (a.length == 0 || a[0].length == 0) {
            throw new RuntimeException("array must be nonempty in both dimensions");
        }

        // double thetaResolutionDegrees = 360.0 / a[0].length;

        VectorFieldMap vectorFieldMap = new VectorFieldMap();

        return new RadialVectorField(vectorFieldMap);
    }

    public static double[] getRadialUnitVector(double[] centerXY, double[] towardXY) {
        double[] fullVector = new double[] {towardXY[0] - centerXY[0], towardXY[1] - centerXY[1]};
        double magnitude = Math.sqrt(Math.pow(fullVector[0], 2) + Math.pow(fullVector[1], 2));

        return new double[] {fullVector[0] / magnitude, fullVector[1] / magnitude};
    }
}


class TornadoField {
    public static void main(String[] args) {
        ;
    }
}
using CSV, DataFrames

@enum Forcing daymet=1 maurer=2 nldas=3

function read_camels_met(id::Int, hru::Int, ds::Forcing)
    if ds == daymet
        metpath = "../Lstm/data/basin_dataset_public_v1p2/basin_mean_forcing/daymet"
        metfile = @sprintf("%s/%02d/%08d_lump_cida_forcing_leap.txt", metpath, hru, id)
    elseif ds == maurer
        metpath = "../Lstm/data/basin_dataset_public_v1p2/basin_mean_forcing/maurer"
        metfile = @sprintf("%s/%02d/%08d_lump_maurer_forcing_leap.txt", metpath, hru, id)
    else
        metpath = "../Lstm/data/basin_dataset_public_v1p2/basin_mean_forcing/nldas"
        metfile = @sprintf("%s/%02d/%08d_lump_nldas_forcing_leap.txt", metpath, hru, id)
    end
    header = ["Year", "Mnth", "Day", "Hr", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
    m = CSV.File(metfile, delim="\t", skipto=5, header=header, normalizenames=true) |> DataFrame
    m[!, :Date] = Date.(m[!, :Year], m[!, :Mnth], m[!, :Day])
    m
end

function read_camels_q(id::Int, hru::Int)
    qpath = "../Lstm/data/basin_dataset_public_v1p2/usgs_streamflow"
    qfile = @sprintf("%s/%02d/%08d_streamflow_qc.txt", qpath, hru, id)
    q = CSV.File(qfile, delim=" ", header=0, missingstring="-999.00") |> DataFrame
    q[!, :Date] = Date.(q[!, 2], q[!, 3], q[!, 4])
    q
end

function read_camels(id::Int, hru::Int, ds::Forcing=daymet)
    m = read_camels_met(id, hru, ds)
    q = read_camels_q(id, hru)
    out = innerjoin(m[!, [6, 7, 9, 10, 11, 12]], q[!, [5, 7]], on= :Date)
    out
end

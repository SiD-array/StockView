import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function VolumeChartCard({ chartData }) {
  return (
    <div className="bg-white rounded-2xl shadow p-4 w-full xl:w-[48%]">
      <h2 className="text-lg font-semibold mb-2">Trading Volume</h2>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tick={{ fontSize: 10 }} />
          <YAxis hide />
          <Tooltip />
          <Bar dataKey="volume" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
